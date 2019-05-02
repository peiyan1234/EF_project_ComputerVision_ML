import json
import os
import glob
import math
import numpy as np
from itertools import combinations
from PIL import Image, ImageDraw

"""
This tool is developed by Alvin Pei-Yan, Li.
Email: Alvin.Li@acer.com / d05548014@ntu.edu.tw
"""

def preprocessing(filename_png_dir):
    """Pre-processing

    Arg:
      filename_png_dir: a string of the absolute path of a .png file.

    Return:
      an modified image
    """

    img = Image.open(filename_png_dir)
    img = np.asarray(img)
    
    img = histogram_equalization(img)
    enhanced_img = Image.fromarray(np.uint8(img)) 
    enhanced_img.show()

    img = median_filter(img) 
    reduced_img = Image.fromarray(np.uint8(img)) 
    reduced_img.show()

    img, covered_img = Sobel_edge_detection(img)
    edge_img = Image.fromarray(np.uint8(img)) 
    edge_img.show()

    covered_img = Image.fromarray(np.uint8(covered_img)) 
    covered_img.show()

def histogram_equalization(img):
    """An contrast enhancer

    Arg:
      img: an ultrasound image represented in a numpy array.

    Return:
      enhanced_img: an enhanced image
    """

    img_shape = img.shape
    flattened_img = img.flatten()
    enhanced_img = np.zeros(flattened_img.shape)

    L = 256 # the number of pixel intensity levels

    histogram = np.zeros(L)
    for pixel in flattened_img:
        histogram[pixel] += 1

    histogram = histogram / 3
    cumulative_sum = np.cumsum(histogram)
    ImgTransformation = (L - 1) * (cumulative_sum - min(cumulative_sum)) / (max(cumulative_sum - min(cumulative_sum))) 

    enhanced_img = ImgTransformation[flattened_img]
    enhanced_img = np.reshape(enhanced_img, img_shape)

    return enhanced_img

def median_filter(img):
    """Reduce the effect of speckle noise

    Arg:
      img: an enhanced image.

    Return:
      noise_reduced_img: a noise reduced image
    """
    img_width, img_height, img_depth = img.shape
    noise_reduced_img = np.zeros(img.shape)

    for w in range(4, img_width-4):
            for h in range(4, img_height-4):
                    sort_arr = np.sort(img[w-4:w+5, h-4:h+5, 0].flatten())
                    noise_reduced_img[w, h, 0] = sort_arr[40]

    noise_reduced_img[:, :, 1] = noise_reduced_img[:, :, 0]
    noise_reduced_img[:, :, 2] = noise_reduced_img[:, :, 0]

    return noise_reduced_img

def Sobel_edge_detection(img):
    """Reduce the effect of speckle noise

    Arg:
      img: an image.

    Return:
      edge_img: the edge image of img
    """
    img_width, img_height, img_depth = img.shape
    edge_img = np.zeros(img.shape)

    for w in range(1, img_width-1):
        for h in range(1, img_height-1):
            patch_of_img = img[w-1:w+2, h-1:h+2, 0].flatten()
            Gx = np.dot([1, 0, -1, 2, 0, -2, 1, 0, -1], patch_of_img)
            Gy = np.dot([1, 2, 1, 0, 0, 0, -1, -2, -1], patch_of_img)
            edge_img[w, h, 0] = np.sqrt(Gx*Gx + Gy*Gy)

    edge_img[:, :, 1] = edge_img[:, :, 0]
    edge_img[:, :, 2] = edge_img[:, :, 0]

    covered_img = img + edge_img

    return edge_img, covered_img
