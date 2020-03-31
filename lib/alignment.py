import numpy as np 
import cv2
from matplotlib import pyplot as plt 

def alignment(images, std_img):
    images_move = np.zeros((len(images), 2))
    neighbors = [[0, 0], [0, -1], [0, 1], [-1, 0], [-1, -1], [-1, 1], [1, 0], [1, -1], [1, 1]]
    for image in images:
        image_resize = [image]
        for i in range(6):
            print('resizing image no.{} ...'.format(len(image_resize)))
            image_resize.append( cv2.resize(image_resize[-1], dsize = (0,0), fx = 2, fy = 2) )
            cv2.imwrite('test{}.jpg'.format(i), image_resize[-1])
            for pos in neighbors:
                print('calaulating position {}'.format(pos))
    return images

            
