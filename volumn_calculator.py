import json
import os
import glob
import numpy as np
from PIL import Image

"""
This tool is developed by Alvin Pei-Yan, Li.
Please inform him for the authorizatiion before further utilities.
Email: Alvin.Li@acer.com / d05548014@ntu.edu.tw
"""

vertices_with_max_area = [] # a list of three points whose traingle area is the largest
apex_cordis = [] # the coordinates of an apex cordis

def LV_volumn_calculator(filename_json_dir):
    """
    This function calculate the volumn of a left ventricle.
    """
    labels_list = get_labels(filename_json_dir)
    find_the_maxarea_triangle(labels_list)
    find_the_apex_cordis(ventrics_with_max_area)
    LV_volumn = Modified_Simpson_calculator(labels_list)

    return LV_volumn

def get_labels(filename_json_dir):
    """
    This function outputs the coordinations of labels on an ultrasound image
    """
    filename = glob.glob(filename_json_dir + '\\' + '*.json')
    with open(filename, encoding='utf-8') as json_file:
        label_dic = json.load(json_file)

    return label_dic['shapes'][0]['points']

def find_the_maxarea_triangle(labels_list):
    """
    """
    Area = 0



def find_the_apex_cordis(ventrics_with_max_area):
    """
    """


def Modified_Simpson_calculator(labels_list):
    """
    """
    return LV_volumn
