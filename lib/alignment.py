import numpy as np 
import cv2
from matplotlib import pyplot as plt 

def alignment(images, std_img):
    images_move = np.zeros((len(images), 2))
    neighbors = [[0, 0], [0, -1], [0, 1], [-1, 0], [-1, -1], [-1, 1], [1, 0], [1, -1], [1, 1]]
    for image in images:
        for i in range(6, -1, -1):
            for pos in neighbors:
                image_resize = cv2.resize(image, fx = 2**i, fy = 2**i)
    return images

            
