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

def main():

    dir_whereamI = os.getcwd()
    for filename in glob.glob('*.png'):
        modified_img = preprocessing(filename)
        savingdir = os.path.join(dir_whereamI, filename.replace(".png","_enhanced.png"))
        modified_img.save(savingdir)


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
    #enhanced_img.show()

    img = median_filter(img) 
    reduced_img = Image.fromarray(np.uint8(img)) 
    #reduced_img.show()

    #img = Kapur_entropy_segmentation(img)
    #Kapur_img = Image.fromarray(np.uint8(img)) 
    #Kapur_img.show()

    img, covered_img = Sobel_edge_detection(img)
    edge_img = Image.fromarray(np.uint8(img)) 
    #edge_img.show()

    covered_img = Image.fromarray(np.uint8(covered_img)) 
    #covered_img.show()

    original_img = Image.open(filename_png_dir)
    edged_img = Image.fromarray(np.uint8(img + original_img))
    #edged_img.show()
    return edged_img

def histogram_equalization(img):
    """An contrast enhancer

    Arg:
      img: an ultrasound image represented in a numpy array.

    Return:
      enhanced_img: an enhanced image
    """
    img_width, img_height, img_depth = img.shape

    L = 256 # the number of pixel intensity levels
    enhanced_img = np.zeros(img.shape)
    
    list_pixelvalues = np.unique(img[:, :, 0].flatten()) 

    histogram = np.zeros(L)
    basket = 0
    for pixel in list_pixelvalues:
        histogram[basket] = np.count_nonzero(img[:, :, 0] == pixel)
        basket += 1
        if basket > L-1:
            break

    cumulative_sum = np.cumsum(histogram)
    for index in range(0, len(list_pixelvalues)):
        pixel = list_pixelvalues[index]
        enhanced_img[img == pixel] = (L - 1) * (cumulative_sum[index] - min(cumulative_sum)) / (max(cumulative_sum - min(cumulative_sum))) 

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

def Kapur_entropy_segmentation(img):
    """Reduce the effect of speckle noise

    Arg:
      img: a noise reduced image.

    Return:
      Kapur_img: a binary image.
    """
    
    Kapur_img = np.zeros(img.shape)

    L = 256 # the number of pixel intensity levels
    img_slice = img[:, :, 0]
    list_pixelvalues = np.unique(img_slice.flatten()) 
    N = len(img_slice.flatten()) # the total number of pixels

    histogram = np.zeros(L)
    basket = 0
    for pixel in list_pixelvalues:
        histogram[basket] = np.count_nonzero(img_slice == pixel)
        basket += 1
        if basket > L-1:
            break

    histogram = histogram / N

    optimizer = 0 # initialize the optimizer
    Hn = 0
    for index in range(0, len(list_pixelvalues)):
        if histogram[index] != 0:
            Hn = Hn + (-1 * histogram[index] * np.log(histogram[index]))

    Ps = 0 #histogram[0]
    Hs = 0 #( -1 * histogram[0] * np.log(histogram[0]) )
    optimizer = 0 #list_pixelvalues[0]
    Phi = 0 #np.log(Ps * (1 - Ps)) + Hs / Ps + (Hn - Hs) / (1 - Ps)

    for index in range(0, len(list_pixelvalues)):        
        if histogram[index] != 0:   
            Ps = Ps + histogram[index]
            if Ps < 1:
                Hs = Hs + ( -1 * histogram[index] * np.log(histogram[index]) )
                value = np.log(Ps * (1 - Ps)) + Hs / Ps + (Hn - Hs) / (1 - Ps)
                optimizer = list_pixelvalues[index] * (value > Phi) + optimizer * (value <= Phi)
                Phi = value * (value > Phi) + Phi * (value <= Phi)

    Kapur_img[img < optimizer] = 0
    Kapur_img[img >= optimizer] = 255

    return Kapur_img


if __name__ == '__main__':
    main()