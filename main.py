import os
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from matplotlib import pyplot as plt

''' Some configs. '''
IMAGE_DIR = 'images/set_1/jpg'
RESULT_DIR = 'results'
image_files = sorted(os.listdir(IMAGE_DIR))

def get_labeled_exif(exif):
    labeled = {}
    for (key, val) in exif.items():
        labeled[TAGS.get(key)] = val
    return labeled

def read_image(file_name, image_dir = IMAGE_DIR, scale = None):
    file_path = os.path.join(image_dir, file_name)
    print("Reading image {} ...".format(file_path))
    img = Image.open(file_path)
    h, w = img.size
    exifs = get_labeled_exif(img._getexif())
    exposure_sec, exposure_base = exifs['ExposureTime']
    exposure_time = exposure_sec * 1.0 / exposure_base
    if scale:
        img = img.resize((h // scale, w // scale), Image.LANCZOS)
    return np.asarray(img), exposure_time

def get_rgb_channels(image):
    ''' Image dimension is (H, W, C), where C is in RGB order '''
    return image[:,:,0], image[:,:,1], image[:,:,2] 

def sample_pixels(h, w, x = 20, y = 20):
    ''' 
        Sample pixel positions in a h * w image. 

        Returns a list of tuples representing pixel positions.
    '''
    pos = []
    h_step, w_step = h // (x + 1), w // (y + 1)
    for i in range(1, x+1):
        for j in range(1, y+1):
            pos.append((i * h_step, j * w_step))
    return pos
        
def z_weights(zmin = 0, zmax = 255):
    zmid = (zmin + zmax) // 2
    def hat(z):
        return z - zmin if z <= zmid else zmax - z
    return np.array([hat(z) + 1 for z in range(zmin, zmax + 1)], dtype = np.float32)

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
        A[k, i - 1] = l * w[i]
        A[k, i] = -2 * l * w[i]
        A[k, i + 1] = l * w[i]
        k += 1
    x = np.linalg.lstsq(A, b, rcond = None)[0].ravel()
    g = x[:256]
    E = x[256:]
    return g, E

def get_radiance_map(images, g, exp, w):
    print("Processing radiance map ...")
    _h, _w = images[0].shape
    images = np.array(images)
    E = []
    for i, img in enumerate(images):
        E.append(g[img] - exp[i])
    rad = np.average(E, axis=0, weights=w[images])
    return rad

images, exposures = [], []
for f in image_files:
    image, exposure = read_image(f, scale = 5)
    images.append(image)
    exposures.append(exposure)

if __name__ == '__main__':
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    image_height, image_width, _ = images[0].shape
    pixel_positions = sample_pixels(image_height, image_width)

    l = 20
    w = z_weights()

    b = np.log(np.array(exposures, dtype = np.float))
    z = get_z([img[:,:,0] for img in images], pixel_positions)
    g1, E = solve_debevec(z, b, w, l)
    z = get_z([img[:,:,1] for img in images], pixel_positions)
    g2, E = solve_debevec(z, b, w, l)
    z = get_z([img[:,:,2] for img in images], pixel_positions)
    g3, E = solve_debevec(z, b, w, l)

    r_map_r = get_radiance_map([img[:,:,0] for img in images], g1, b, w)
    r_map_g = get_radiance_map([img[:,:,1] for img in images], g2, b, w)
    r_map_b = get_radiance_map([img[:,:,2] for img in images], g3, b, w)
    r_map = np.transpose(np.exp((np.concatenate( ([r_map_b], [r_map_g], [r_map_r]), axis = 0))), (1,2,0))

    cv2.imwrite(os.path.join(RESULT_DIR, 'test.hdr'), r_map.astype(np.float32))

    plt.plot(g1, np.arange(0, 256), 'r', g2, np.arange(0, 256), 'g', g3, np.arange(0, 256), 'b')
    plt.savefig(os.path.join(RESULT_DIR, 'exposure.png'))
