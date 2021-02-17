import cv2
import numpy as np
import random
from random import randint
import os
import glob


###
# Francisco's method to augment glare images
import skimage
from skimage.filters import gaussian

def normalize_image(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def add_glare(background, alpha=1, beta=0.5, gamma=0, severity=1):
    # Generate random glare mask
    # shape = (10,4)
    shape = background.shape

    # layer = np.random.normal(size=(shape[0], shape[1]), loc=0.65, scale=0.3)
    layer = np.random.normal(size=(shape[0], shape[1]), loc=0.65, scale=0.3)
    # h, w = background.shape[0], background.shape[1]
    # layer = cv2.resize(layer, (w, h))
    layer = gaussian(layer, sigma=(np.random.randint(3, 9), np.random.randint(3, 9)))
    layer = normalize_image(layer)
    shape_2 = (shape[0], shape[1], 4)
    layer2 = np.ones(shape_2)*255
    # layer = np.asarray(layer, dtype=np.uint8)
    # Crop to 20x514
    y0 = np.random.randint(0, 80)
    # layer = layer[y0:y0+20, :]
    # Add glare mask to background
    # glared_image = cv2.addWeighted(background, alpha, layer, beta, gamma)
    # layer2 = cv2.cvtColor(layer2, cv2.COLOR_RGB2RGBA).copy()
    glared_image = overlay_image_alpha(background, layer2, (0, 0), layer)
    return glared_image, layer


def augment_glare(img_dir, output_dir):
    img_list = glob.glob(img_dir + '/*.jpg')
    img_list.extend(glob.glob(img_dir + '/*.png'))

    for img in img_list:
        im = cv2.imread(img)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2RGBA).copy()
        gl, _ = add_glare(im)
        bname = os.path.basename(img)
        cv2.imwrite(os.path.join(output_dir, bname), gl)





if __name__ == '__main__':
    img_dir = "test_normal"
    outp_dir = "test_glare_augm"
    augment_glare(img_dir, outp_dir)



