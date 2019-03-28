import json
import os
import glob
import math
import numpy as np
from itertools import combinations
from PIL import Image

"""
This tool is developed by Alvin Pei-Yan, Li.
Please inform him for the authorizatiion before further utilities of this tool.
Email: Alvin.Li@acer.com / d05548014@ntu.edu.tw
"""

vertices_with_max_area = [] # a list of three points whose traingle area is the largest
apex_cordis = [] # the coordinates of an apex cordis

def LV_volumn_calculator(filename_json_dir):
    """
    This function calculate the volumn of a left ventricle.
    """
    labels_list = get_labels(filename_json_dir)
    vertices_with_max_area = find_the_largest_triangle(labels_list)
    apex_cordis = find_the_apex_cordis()
    LV_volumn = Modified_Simpson_calculator(labels_list)

    return LV_volumn

def get_labels(filename_json_dir):
    """
    This function outputs the coordinations of labels on an ultrasound image
    """
    with open(filename_json_dir, encoding='utf-8') as json_file:
        label_dic = json.load(json_file)

    return label_dic['shapes'][0]['points']

def find_the_largest_triangle(labels_list):
    """
    """
    Area = 0
    comb_sets = combinations(labels_list, 3)
    for vertices in comb_sets:
        label1, label2, label3 = vertices
        Area_init = get_triangle_area(label1, label2, label3)
        if Area_init > Area :
            Area = Area_init
            vertices_with_max_area = [label1, label2, label3]

    return vertices_with_max_area

def find_the_apex_cordis():
    """
    """
    label1, label2, label3 = vertices_with_max_area
    x1, y1 = label1 # point1: (x1, y1)
    x2, y2 = label2 # point2: (x2, y2)
    x3, y3 = label3 # point3: (x3, y3)

    distance12 = math.sqrt( (x1 - x2)**2 + (y1 - y2)**2 )
    distance23 = math.sqrt( (x2 - x3)**2 + (y2 - y3)**2 )
    distance13 = math.sqrt( (x1 - x3)**2 + (y1 - y3)**2 )

    condition_a = (distance12 + distance23 > distance13)
    condition_b = (distance12 + distance13 > distance23)
    condition_c = (distance23 + distance13 > distance12)

    if condition_a:
        return label2
    elif condition_b:
        return label1
    elif condition_c:
        return label3

def Modified_Simpson_calculator(labels_list):
    """
    """
    return LV_volumn

def get_triangle_area(label1, label2, label3):

    x1, y1 = label1 # point1: (x1, y1)
    x2, y2 = label2 # point2: (x2, y2)
    x3, y3 = label3 # point3: (x3, y3)

    Array_x = np.asarray([x1, x1, x1, x2, x2, x2, x3, x3, x3])
    Array_y = np.asarray([y1, y2, y3, y1, y2, y3, y1, y2, y3])
    Array_0 = np.asarray([0 ,  1, -1, -1,  0,  1,  1, -1,  0])

    return np.absolute( 0.5 * np.sum( np.multiply( Array_x, np.multiply(Array_y,Array_0) ) ) )
