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

def read_image(file_name, image_dir = IMAGE_DIR):
    file_path = os.path.join(image_dir, file_name)
    print(file_path)
    img = Image.open(file_path)
    exifs = get_labeled_exif(img._getexif())
    exposure_sec, exposure_base = exifs['ExposureTime']
    return {
        'img': np.asarray(img), 
        'exposure': exposure_sec * 1.0 / exposure_base
    }

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
        
images = list(map(read_image, image_files))
image_height = images[0]['img'].shape[0]
image_width = images[0]['img'].shape[1]
pixel_positions = sample_pixels(image_height, image_width)

    