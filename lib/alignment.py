import numpy as np 
import cv2
from matplotlib import pyplot as plt 

def image_to_boolean(image, percentile=50, print=False):
    image_rbg_sum = np.sum(image, axis=2)
    image_threshold = np.percentile(image_rbg_sum, percentile)
    image_boolean = ( image_rbg_sum >= image_threshold )
    if print :
        h, w = image_boolean.shape
        image_test = np.zeros((h, w, 3))
        image_test[:,:,0] = image_boolean*255
        image_test[:,:,1] = image_boolean*255
        image_test[:,:,2] = image_boolean*255
        cv2.imwrite('test.jpg', image_test.astype(np.float32))
    return image_boolean

def image_diff(image_bool1, image_bool2):
    return np.sum( np.logical_xor( image_bool1, image_bool2) )
def image_shift(image, pos, print=False):
    h, w, _ = image.shape
    x, y = pos
    M = np.float32([[1,0,x],[0,1,y]])
    shifted_image = cv2.warpAffine(image, M, (w, h))
    if print :
        cv2.imwrite('test_shift{}.jpg'.format([x,y]), shifted_image[:,:,::-1])
    return shifted_image

def alignment(images, std_img, align_num):
    std_img_booleans = [image_to_boolean(std_img)]
    std_img_resize = [std_img]
    best_shifted_images = []
    for i in range(1,align_num):
        std_img_resize.append( cv2.resize(std_img_resize[-1], dsize=(0,0), fx=0.5, fy=0.5) )
        std_img_booleans.append( image_to_boolean( std_img_resize[-1]) )

    neighbors = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [-1, -1], [-1, 1], [1, 0], [1, -1], [1, 1]])
    for image in images:
        image_resize = [image]
        for i in range(1,align_num):
            image_resize.append( cv2.resize(image_resize[-1], dsize=(0,0), fx=0.5, fy=0.5) )
        
        best_pos = np.array([0, 0])
        for i in range(align_num-1, -1, -1): # for each size
            step = 2**i
            min_diff_neighbor = np.array(neighbors[0])
            min_diff = image.shape[0]*image.shape[1]
            for pos in neighbors:
                shifted_image = image_shift(image_resize[ i ], np.add(best_pos/step, pos) )
                shifted_image_booleans = image_to_boolean( shifted_image)
                diff = image_diff( std_img_booleans[i], shifted_image_booleans)
                if diff < min_diff:
                    min_diff = diff
                    min_diff_neighbor = pos
            best_pos = np.add(best_pos, min_diff_neighbor*step)
        best_shifted_images.append( image_shift(image, best_pos) )
    return best_shifted_images

            
