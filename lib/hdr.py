import os
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from matplotlib import pyplot as plt
from . import hat_functions as hat_func
from . import alignment as align

def get_labeled_exif(exif):
    labeled = {}
    for (key, val) in exif.items():
        labeled[TAGS.get(key)] = val
    return labeled

def read_image(file_name, image_dir, scale = None):
    file_path = os.path.join(image_dir, file_name)
    print("Reading image {} ...".format(file_path))
    img = Image.open(file_path)
    h, w = img.size
    exifs = get_labeled_exif(img._getexif())
    exposure_sec, exposure_base = exifs['ExposureTime']
    exposure_time = exposure_sec * 1.0 / exposure_base
    if scale:
        img = img.resize((int(h / scale), int(w / scale)), Image.LANCZOS)
    return np.asarray(img), exposure_time

def sample_pixels(h, w, x = 20, y = 20):
    ''' 
        Sample pixel positions in a h * w image. 

        Returns a list of tuples representing pixel positions.
    '''
    pos = []
    h_step, w_step = h // (x + 1), w // (y + 1)
    for i in range(1, x + 1):
        for j in range(1, y + 1):
            pos.append((i * h_step, j * w_step))
    return pos
        
def z_weights(zmin = 0, zmax = 255, hat = 'linear'):
    f = None
    if hat == 'none':
        f = lambda z: z
    if hat == 'linear':
        f = hat_func.linear_hat(zmin, zmax)
    elif hat == 'sin':
        f = hat_func.sin_hat(zmin, zmax)
    elif hat == 'gaussian':
        f = hat_func.gaussian_hat(zmin, zmax)
    return np.array([f(z) for z in range(zmin, zmax + 1)], dtype = np.float32)

def get_z(images, pixel_positions):
    ''' Images should be a list of 1-channel (R / G / B) images. '''
    h, w = images[0].shape
    z = np.zeros((len(pixel_positions), len(images)), dtype = np.uint8)
    for i, (x, y) in enumerate(pixel_positions):
        for j, img in enumerate(images):
            z[i, j] = img[x, y]
    return z

def solve_debevec(z, exp, w, l = 5):
    '''
        Algorithm from Debevec 1997.
        "Recovering High Dynamic Range Radiance Maps from Photographs".
    '''
    n, p = z.shape
    assert(len(exp) == p and len(w) == 256)
    A = np.zeros((n * p + 1 + 254, 256 + n))
    b = np.zeros((A.shape[0], 1))
    k = 0
    for i, row in enumerate(z):
        for j, z_ij in enumerate(row):
            w_ij = w[z_ij] # weighted pixel value
            A[k, z_ij] = w_ij
            A[k, i + 256] = -w_ij
            b[k, 0] = w_ij * exp[j]
            k += 1
    A[k, 127] = 1
    k += 1
    for i in range(1, 255): # iterate from 1 to 254
        A[k, i - 1 : i + 2] = l * w[i] * np.array([1, -2, 1])
        k += 1
    x = np.linalg.lstsq(A, b, rcond = None)[0].ravel()
    return x[:256] # return only g(0) ~ g(255)

def get_radiance_map(images, g, exp, w):
    print("Processing radiance map ...")
    _h, _w = images[0].shape
    images = np.array(images)
    E = []
    for i, img in enumerate(images):
        E.append(g[img] - exp[i])
    rad = np.average(E, axis=0, weights=w[images])
    return rad

def print_radiance_map(rads, result_dir='.', colors = ['r', 'g', 'b']):
    for i, rad in enumerate(rads):
        plt.figure()
        plt.imshow(rad, cmap = plt.cm.jet )
        plt.colorbar()
        plt.savefig(os.path.join(result_dir, 'radiance_map_{}.png'.format(colors[i])))
        plt.close()

def hdr(image_dir, result_dir, hat_type, l, scale):
    w = z_weights(hat = hat_type)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    image_files = sorted(os.listdir(image_dir))
    images, exposures = [], []
    for f in image_files:
        image, exposure = read_image(f, image_dir, scale = scale)
        images.append(image)
        exposures.append(exposure)
    
    images = align.alignment(images, images[len(images)//2], 6)
    b = np.log(np.array(exposures, dtype = np.float32))

    image_height, image_width, _ = images[0].shape
    pixel_positions = sample_pixels(image_height, image_width)
    radiance_maps = []
    colors = ['r', 'g', 'b']
    plt.figure()
    for i in range(2, -1, -1): # Process in BGR channel order
        channels = [img[:,:,i] for img in images] # ith channel of each image
        z = get_z(channels, pixel_positions)
        g = solve_debevec(z, b, w, l) # retrieve mapping function
        r = get_radiance_map(channels, g, b, w ) # retrieve channel-wise radiance map
        radiance_maps.append(r)
        plt.plot(g, range(256), colors[i])
    print_radiance_map(radiance_maps, result_dir, colors)
    r_map = np.transpose(np.exp(np.stack(radiance_maps)), (1, 2, 0))
    cv2.imwrite(os.path.join(result_dir, 'result.hdr'), r_map.astype(np.float32))
    plt.savefig(os.path.join(result_dir, 'exposure.png'))
