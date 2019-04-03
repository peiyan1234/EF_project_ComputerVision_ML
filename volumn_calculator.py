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

global apex_cordis # the coordinates of an apex cordis
global base_vertices # a list of three points that will be used in Modified Simpson's Method later.

def Modified_Simpson_calculator(filename_json_dir):
    """
    This function calculate the volumn of a left ventricle.
    """
    labels_list = get_labels(filename_json_dir)
    largets_triangle = find_the_largest_triangle(labels_list)
    base_vertices = check_the_bottom_2_points(labels_list, largets_triangle)

    x_ap, y_ap = base_vertices[1]
    x_mid = ( base_vertices[2][1] + base_vertices[3][1] ) / 2.0
    y_mid = ( base_vertices[2][2] + base_vertices[3][2] ) / 2.0
    
    N = 20 # the amount of segmented lines

    list_segmented_N_x = list(np.linspace(x_ap, x_mid, N+1))
    list_segmented_N_y = list(np.linspace(y_ap, x_mid, N+1))

    height = math.sqrt( (x_ap - x_mid)**2 + (y_ap - y_mid)**2 ) / N

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
    largets_triangle = []

    comb_sets = combinations(labels_list, 3)
    for vertices in comb_sets:
        label1, label2, label3 = vertices
        Area_init = get_triangle_area(label1, label2, label3)
        if Area_init > Area :
            Area = Area_init
            largets_triangle = [label1, label2, label3]

    return largets_triangle # a list of three points whose traingle area is the largest

def get_triangle_area(label1, label2, label3):

    x1, y1 = label1 # point1: (x1, y1)
    x2, y2 = label2 # point2: (x2, y2)
    x3, y3 = label3 # point3: (x3, y3)

    Array_x = np.asarray([x1, x1, x1, x2, x2, x2, x3, x3, x3])
    Array_y = np.asarray([y1, y2, y3, y1, y2, y3, y1, y2, y3])
    Array_0 = np.asarray([0 ,  1, -1, -1,  0,  1,  1, -1,  0])

    return np.absolute( 0.5 * np.sum( np.multiply( Array_x, np.multiply(Array_y,Array_0) ) ) )

def check_the_bottom_2_points(labels_list, largets_triangle):

	label1, label2, label3 = largets_triangle

	x1, y1 = label1 # point1: (x1, y1)
	x2, y2 = label2 # point2: (x2, y2)
	x3, y3 = label3 # point3: (x3, y3)

	apex_cordis = find_the_apex_cordis(largets_triangle)

	K_numerator = (x2 + x3 - 2*x_ap) * (label1 == apex_cordis) + \
    			  (x1 + x3 - 2*x_ap) * (label2 == apex_cordis) + \
    			  (x1 + x2 - 2*x_ap) * (label3 == apex_cordis)

	K_denominator = (y2 + y3 - 2*y_ap) * (label1 == apex_cordis) + \
    				(y1 + y3 - 2*y_ap) * (label2 == apex_cordis) + \
    				(y1 + y2 - 2*y_ap) * (label3 == apex_cordis)

    # return a list of three points that will be used in Modified Simpson's Method later.
    return get_corrected_base_vertices(K_numerator, K_denominator, labels_list)

def find_the_apex_cordis(largets_triangle):
    """
    """
    label1, label2, label3 = largets_triangle
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

def get_corrected_base_vertices(K_numerator, K_denominator, labels_list):

	if K_denominator == 0:
 		print('Got a wrong image!')
 		break

	else:
 		x_ap, y_ap = apex_cordis
 		x_bleft, y_bleft = apex_cordis # set the point on the bottom left for initialization
 		x_bright, y_bright = apex_cordis # set the point on the bottom right for initialization

 		K = K_numerator / K_denominator
 		C = -K*y_ap + x_ap

		for label in labels_list:
 			x_label, y_label = label
 			distance_label_2_apex_cordis = math.sqrt( (x_label - x_ap)**2 + (y_label - y_ap)**2 )
 			distance_bleft_2_apex_cordis = math.sqrt( (x_bleft - x_ap)**2 + (y_bleft - y_ap)**2 )
 			distance_bright_2_apex_cordis = math.sqrt( (x_bright - x_ap)**2 + (y_bright - y_ap)**2 )

 			check_left = ( distance_label_2_apex_cordis > distance_bleft_2_apex_cordis ) * ( x_label < (K*y_label + C) )
 			x_bleft = x_bleft*(1 - check_left) + x_label*check_left
 			y_bleft = y_bleft*(1 - check_left) + y_label*check_left

 			check_right = ( distance_label_2_apex_cordis > distance_bright_2_apex_cordis ) * ( x_label > (K*y_label + C) )
 			x_bright = x_bright*(1 - check_right) + x_label*check_right
 			y_bright = y_bright*(1 - check_right) + y_label*check_right
	
		return [apex_cordis, [x_bleft, y_bleft], [x_bright, y_bright]]