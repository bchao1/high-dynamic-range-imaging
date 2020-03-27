import os
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

''' Some configs. '''
IMAGE_DIR = 'images/set_1/jpg'
image_files = sorted(os.listdir(IMAGE_DIR))

def get_labeled_exif(exif):
    labeled = {}
    for (key, val) in exif.items():
        labeled[TAGS.get(key)] = val
    return labeled

def read_image(file_name, image_dir = IMAGE_DIR, resize = False):
    file_path = os.path.join(image_dir, file_name)
    print("Reading image {} ...".format(file_path))
    img = Image.open(file_path)
    exifs = get_labeled_exif(img._getexif())
    exposure_sec, exposure_base = exifs['ExposureTime']
    return np.asarray(img), exposure_sec * 1.0 / exposure_base

def get_rgb_channels(image):
    ''' Image dimension is (H, W, C), where C is in RGB order '''
    return image[:,:,0], image[:,:,1], image[:,:,2] 

def sample_pixels(h, w):
    ''' Sample pixel positions in a h * w image. '''
    # Simple equidistant 5 * 10 sampling at this time. Can be modified.
    pos = []
    h_step, w_step = h // (5 + 1), w // (10 + 1)
    for i in range(1, 6):
        for j in range(1, 11):
            pos.append((i * h_step, j * w_step))
    return pos
        
def z_weights(zmin = 0, zmax = 255):
    zmid = (zmin + zmax) // 2
    def hat(z):
        return z - zmin if z <= zmin else zmax - z
    return np.array([hat(z) for z in range(zmin, zmax + 1)], dtype = np.float)

def get_z(images, pixel_positions):
    ''' Images should be a list of 1-channel (R / G / B) images. '''
    h, w = images[0].shape
    z = np.zeros((len(pixel_positions), len(images)), dtype = np.float)
    for i, (x, y) in enumerate(pixel_positions):
        for j, img in enumerate(images):
            z[i, j] = img[x, y]
    return z


images, exposures = [], []
for f in image_files:
    image, exposure = read_image(f)
    images.append(image)
    exposures.append(exposure)

image_height, image_width, _ = images[0].shape

pixel_positions = sample_pixels(image_height, image_width)

l = 1
w_z = z_weights()
b = np.array(exposures, dtype = np.float)
z = get_z([img[:,:,0] for img in images], pixel_positions)
print(z.shape)