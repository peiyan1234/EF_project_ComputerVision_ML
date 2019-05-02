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

global apex_cordis # the coordinate of an apex cordis
global base_vertices # a list of three points that will be used in Modified Simpson's Method later.
N = 20 # the amount of disks for discretizely summing up the LV volume

def Modified_Simpson_calculator(filename_json_dir):
    """Get LV volume by Modified Simpson's Method

    Arg:
      filename_json_dir: a string of the absolute path of a .json file.

    Return:
      Left Ventricle (LV) volume
    """

    labels_list = get_labels(filename_json_dir)
    largest_triangle = find_the_largest_triangle(labels_list)
    base_vertices = check_the_bottom_2_points(labels_list, largest_triangle)

    return get_LV_volume(base_vertices, N, labels_list)

def get_labels(filename_json_dir):
    """Get a list of the coordinations of labels from the .json file

    Arg:
      filename_json_dir: a string of the absolute path of a .json file.

    Return:
      a list of the coordinations of labels 
    """

    with open(filename_json_dir, encoding='utf-8') as json_file:
        label_dic = json.load(json_file)

    return label_dic['shapes'][0]['points']

def find_the_largest_triangle(labels_list):
    """Find the largest triangle consisting of the apex cordis and other two points

    Arg:
      labels_list: a list of labels from the output of get_labels

    Return:
      largest_triangle: a list of three points whose traingle area is the largest
    """

    Area = 0
    largest_triangle = []

    comb_sets = combinations(labels_list, 3)
    for vertices in comb_sets:
        label1, label2, label3 = vertices
        Area_init = get_triangle_area(label1, label2, label3)
        if Area_init > Area :
            Area = Area_init
            largest_triangle = [label1, label2, label3]

    return largest_triangle 

def get_triangle_area(label1, label2, label3):
    """Calculate the triangle area consisting of three labels

    Args:
      label1, label2, label3: the coordinates of labels

    Return:
      the triangle area of these three labels
    """

    x1, y1 = label1 
    x2, y2 = label2 
    x3, y3 = label3 

    Array_x = np.asarray([x1, x1, x1, x2, x2, x2, x3, x3, x3])
    Array_y = np.asarray([y1, y2, y3, y1, y2, y3, y1, y2, y3])
    Array_0 = np.asarray([0 ,  1, -1, -1,  0,  1,  1, -1,  0])

    return np.absolute( 0.5 * np.sum( np.multiply( Array_x, np.multiply(Array_y,Array_0) ) ) )

def check_the_bottom_2_points(labels_list, largest_triangle):
    """Check the bottom points are correct for Modified Simpson's Method

    Args:
      labels_list: the output of get_labels
      largest_triangle: the output of find_the_largest_triangle

    Return:
      a list of three points that will be used in Modified Simpson's Method later.
    """

    label1, label2, label3 = largest_triangle

    x1, y1 = label1 # point1: (x1, y1)
    x2, y2 = label2 # point2: (x2, y2)
    x3, y3 = label3 # point3: (x3, y3)

    apex_cordis = find_the_apex_cordis(largest_triangle)
    x_ap, y_ap = apex_cordis

    K_numerator = (x2 + x3 - 2*x_ap) * (label1 == apex_cordis) + (x1 + x3 - 2*x_ap) * (label2 == apex_cordis) + (x1 + x2 - 2*x_ap) * (label3 == apex_cordis)
    K_denominator = (y2 + y3 - 2*y_ap) * (label1 == apex_cordis) + (y1 + y3 - 2*y_ap) * (label2 == apex_cordis) + (y1 + y2 - 2*y_ap) * (label3 == apex_cordis)

    return get_corrected_base_vertices(K_numerator, K_denominator, labels_list, apex_cordis)

def find_the_apex_cordis(largest_triangle):
    """Find which point is the apex cordis

    Arg:
      largest_triangle: the output of find_the_largest_triangle

    Return:
      the coordinate of the apex cordis
    """

    label1, label2, label3 = largest_triangle
    x1, y1 = label1 
    x2, y2 = label2 
    x3, y3 = label3 

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

def get_corrected_base_vertices(K_numerator, K_denominator, labels_list, apex_cordis):
    """Get the correct three base points for Modified Simpson's Method

    Args:
      K_numerator: apply to separate the labels to two groups. 
      K_denominator: apply to separate the labels to two groups. 
      labels_list: the output of get_labels
      apex_cordis: the coordinate of the apex cordis

    Return:
      a list of the coordinates of the apex cordis and the left/right bottom points
    """

    if K_denominator == 0:
        print('Got a wrong image!')

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

def get_LV_volume(base_vertices, N, labels_list):
    """Get the Left Ventricle volume 

    Args:
      base_vertices: the output of get_corrected_base_vertices
      N: 20 in typical, the number of discretized disks used to approximate the LV volume by integration 
      labels_list: the output of get_labels

    Return:
      a list of the coordinates of the apex cordis and the left/right bottom points
    """

    list_segmented_N_x, list_segmented_N_y, x_l, y_l, x_L, y_L, height = get_segmented_line(base_vertices, N)
    K, C, K_b1, C_b1, K_b2, C_b2 = get_distinguish_parameters(base_vertices)

    x_ap, y_ap = base_vertices[0]
    x_mid = ( base_vertices[1][0] + base_vertices[2][0] ) / 2.0
    y_mid = ( base_vertices[1][1] + base_vertices[2][1] ) / 2.0

    Left_labels_by_middle, Right_labels_by_middle, Left_labels_b1, Right_labels_b1, Right_labels_b2, Left_labels_b2, count_the_rest_left_labels, count_the_rest_right_labels = get_sides_labels(base_vertices, labels_list)
    
    LV_volume = 0
    for segment in range(0, len(list_segmented_N_x)-1):
        x_h = list_segmented_N_x[segment + 1]
        y_h = list_segmented_N_y[segment + 1]
        
        distance_h_2_apex_cordis = math.sqrt( (x_h - x_ap)**2 + (y_h - y_ap)**2 )
        distance_l_2_apex_cordis = math.sqrt( (x_l - x_ap)**2 + (y_l - y_ap)**2 )
        distance_m_2_apex_cordis = math.sqrt( (x_mid - x_ap)**2 + (y_mid - y_ap)**2 )

        if distance_h_2_apex_cordis <= distance_m_2_apex_cordis:
            x_left1, y_left1 = x_ap, y_ap
            x_left2, y_left2 = x_mid, y_mid
            for label in Left_labels_by_middle:
                x, y = label
                cos_theta_1 = get_cos_theta(x_left1, y_left1, x_ap, y_ap, x_h, y_h)
                cos_theta_2 = get_cos_theta(x_left2, y_left2, x_ap, y_ap, x_h, y_h)
                cos_theta_label = get_cos_theta(x, y, x_ap, y_ap, x_h, y_h)

                x_left1 = x * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + x_left1 * (1- (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))
                y_left1 = y * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + y_left1 * (1- (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))

                x_left2 = x * (cos_theta_label < 0) * (cos_theta_label > cos_theta_2) + x_left2 * (1 - (cos_theta_label < 0) * (cos_theta_label > cos_theta_2))
                y_left2 = y * (cos_theta_label < 0) * (cos_theta_label > cos_theta_2) + y_left2 * (1 - (cos_theta_label < 0) * (cos_theta_label > cos_theta_2))
                
            checker1 = (x_left2 == x_left1) * (y_left2 == y_left1)
            if checker1 == 1:
                x_left = x_left1
                y_left = y_left1
            else:
                t = np.dot([x_h - x_left1, y_h - y_left1], [x_ap - x_h, y_ap - y_h]) / np.dot([x_left2 - x_left1, y_left2 - y_left1], [x_ap - x_h, y_ap - y_h])
                x_left = x_left1 + (x_left2 - x_left1) * t
                y_left = y_left1 + (y_left2 - y_left1) * t

            x_right1, y_right1 = x_ap, y_ap
            x_right2, y_right2 = x_mid, y_mid
            for label in Right_labels_by_middle:
                x, y = label
                cos_theta_1 = get_cos_theta(x_right1, y_right1, x_ap, y_ap, x_h, y_h)
                cos_theta_2 = get_cos_theta(x_right2, y_right2, x_ap, y_ap, x_h, y_h)
                cos_theta_label = get_cos_theta(x, y, x_ap, y_ap, x_h, y_h)

                x_right1 = x * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + x_right1 * (1 - (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))
                y_right1 = y * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + y_right1 * (1 - (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))

                x_right2 = x * (cos_theta_label < 0) * (cos_theta_label > cos_theta_2) + x_right2 * (1 - (cos_theta_label < 0) * (cos_theta_label > cos_theta_2))
                y_right2 = y * (cos_theta_label < 0) * (cos_theta_label > cos_theta_2) + y_right2 * (1 - (cos_theta_label < 0) * (cos_theta_label > cos_theta_2))
            
            checker2 = (x_right2 == x_right1) * (y_right2 == y_right1)
            if checker2 == 1:
                x_right = x_right1
                y_right = y_right1
            else:
                t = np.dot([x_h - x_right1, y_h - y_right1], [x_ap - x_h, y_ap - y_h]) / np.dot([x_right2 - x_right1, y_right2 - y_right1], [x_ap - x_h, y_ap - y_h])
                x_right = x_right1 + (x_right2 - x_right1) * t
                y_right = y_right1 + (y_right2 - y_right1) * t

            if distance_h_2_apex_cordis <= distance_l_2_apex_cordis:
                LV_volume = LV_volume + math.pi * 0.25 * height * ( math.sqrt( (x_left - x_right)**2 + (y_left - y_right)**2 ) )**2
            
            elif (distance_h_2_apex_cordis > distance_l_2_apex_cordis) * (distance_h_2_apex_cordis <= distance_m_2_apex_cordis):   
                distance_left_2_middle = math.sqrt( (x_left - x_h)**2 + (y_left - y_h)**2 )
                distance_right_2_middle = math.sqrt( (x_right - x_h)**2 + (y_right - y_h)**2 )
                checker_longest = (distance_left_2_middle >= distance_right_2_middle)

                LV_volume = LV_volume + (math.pi * height * distance_left_2_middle**2) * checker_longest + (math.pi * height * distance_right_2_middle**2) * (1 - checker_longest)

        elif distance_h_2_apex_cordis > distance_m_2_apex_cordis:
            if count_the_rest_left_labels >= count_the_rest_right_labels:
                Left_labels_rest = []
                Right_labels_rest = []
                for label in Left_labels_by_middle:
                    x_label, y_label = label
                    if x_label < (K_b1*y_label + C_b1):
                        Left_labels_rest.append(label)
                    elif x_label > (K_b1*y_label + C_b1):
                        Right_labels_rest.append(label)
                                
                                
                Left_labels_rest =  Left_labels_rest + [base_vertices[1]]
                Right_labels_rest = Right_labels_rest + [base_vertices[1]]
                
                x_left1, y_left1 = x_mid, y_mid
                x_left2, y_left2 = 2*x_L - x_ap, 2*y_L -y_ap
                for label in Left_labels_rest:
                    x, y = label
                    cos_theta_1 = get_cos_theta(x_left1, y_left1, x_ap, y_ap, x_h, y_h)
                    cos_theta_2 = get_cos_theta(x_left2, y_left2, x_ap, y_ap, x_h, y_h)
                    cos_theta_label = get_cos_theta(x, y, x_ap, y_ap, x_h, y_h)

                    x_left1 = x * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + x_left1 * (1- (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))
                    y_left1 = y * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + y_left1 * (1- (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))

                    x_left2 = x * (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2) + x_left2 * (1 - (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2))
                    y_left2 = y * (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2) + y_left2 * (1 - (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2))
                
                checker1 = (x_left2 == x_left1) * (y_left2 == y_left1)
                if checker1 == 1:
                    x_left = x_left1
                    y_left = y_left1
                else:
                    t = np.dot([x_h - x_left1, y_h - y_left1], [x_ap - x_h, y_ap - y_h]) / np.dot([x_left2 - x_left1, y_left2 - y_left1], [x_ap - x_h, y_ap - y_h])
                    x_left = x_left1 + (x_left2 - x_left1) * t
                    y_left = y_left1 + (y_left2 - y_left1) * t

                x_right1, y_right1 = x_mid, y_mid
                x_right2, y_right2 = 2*x_L - x_ap, 2*y_L -y_ap
                for label in Right_labels_rest:
                    x, y = label
                    cos_theta_1 = get_cos_theta(x_right1, y_right1, x_ap, y_ap, x_h, y_h)
                    cos_theta_2 = get_cos_theta(x_right2, y_right2, x_ap, y_ap, x_h, y_h)
                    cos_theta_label = get_cos_theta(x, y, x_ap, y_ap, x_h, y_h)

                    cos_theta_1 = np.dot([x_right1 - x_h, y_right1 - y_h], [x_ap - x_h, y_ap - y_h]) / ( math.sqrt( (x_right1 - x_h)**2 + (y_right1 - y_h)**2 ) * math.sqrt( (x_ap - x_h)**2 + (y_ap - y_h)**2 ) )
                    cos_theta_2 = np.dot([x_right2 - x_h, y_right2 - y_h], [x_ap - x_h, y_ap - y_h]) / ( math.sqrt( (x_right2 - x_h)**2 + (y_right2 - y_h)**2 ) * math.sqrt( (x_ap - x_h)**2 + (y_ap - y_h)**2 ) )
                    cos_theta_label = np.dot([x - x_h, y - y_h], [x_ap - x_h, y_ap - y_h]) / ( math.sqrt( (x - x_h)**2 + (y - y_h)**2 ) * math.sqrt( (x_ap - x_h)**2 + (y_ap - y_h)**2 ) )

                    x_right1 = x * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + x_right1 * (1 - (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))
                    y_right1 = y * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + y_right1 * (1 - (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))

                    x_right2 = x * (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2) + x_right2 * (1 - (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2))
                    y_right2 = y * (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2) + y_right2 * (1 - (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2))

                checker2 = (x_right2 == x_right1) * (y_right2 == y_right1)
                if checker2 == 1:
                    x_right = x_right1
                    y_right = y_right1
                else:
                    t = np.dot([x_h - x_right1, y_h - y_right1], [x_ap - x_h, y_ap - y_h]) / np.dot([x_right2 - x_right1, y_right2 - y_right1], [x_ap - x_h, y_ap - y_h])
                    x_right = x_right1 + (x_right2 - x_right1) * t
                    y_right = y_right1 + (y_right2 - y_right1) * t

                distance_left_2_middle = math.sqrt( (x_left - x_h)**2 + (y_left - y_h)**2 )
                distance_right_2_middle = math.sqrt( (x_right - x_h)**2 + (y_right - y_h)**2 )

                LV_volume = LV_volume + math.pi * height * (distance_left_2_middle**2 - distance_right_2_middle**2)

            else:
                Left_labels_rest = []
                Right_labels_rest = []
                for label in Right_labels_by_middle:
                    x_label, y_label = label
                    if x_label < (K_b2*y_label + C_b2):
                        Left_labels_rest.append(label)
                    elif x_label > (K_b2*y_label + C_b2):
                        Right_labels_rest.append(label)

                Left_labels_rest = Left_labels_rest + [base_vertices[2]]
                Right_labels_rest = Right_labels_rest + [base_vertices[2]]

                x_left1, y_left1 = x_mid, y_mid
                x_left2, y_left2 = 2*x_L - x_ap, 2*y_L -y_ap
                for label in Left_labels_rest:
                    x, y = label
                    cos_theta_1 = get_cos_theta(x_left1, y_left1, x_ap, y_ap, x_h, y_h)
                    cos_theta_2 = get_cos_theta(x_left2, y_left2, x_ap, y_ap, x_h, y_h)
                    cos_theta_label = get_cos_theta(x, y, x_ap, y_ap, x_h, y_h)

                    x_left1 = x * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + x_left1 * (1- (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))
                    y_left1 = y * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + y_left1 * (1- (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))

                    x_left2 = x * (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2) + x_left2 * (1 - (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2))
                    y_left2 = y * (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2) + y_left2 * (1 - (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2))
                
                checker1 = (x_left2 == x_left1) * (y_left2 == y_left1)
                if checker1 == 1:
                    x_left = x_left1
                    y_left = y_left1
                else:
                    t = np.dot([x_h - x_left1, y_h - y_left1], [x_ap - x_h, y_ap - y_h]) / np.dot([x_left2 - x_left1, y_left2 - y_left1], [x_ap - x_h, y_ap - y_h])
                    x_left = x_left1 + (x_left2 - x_left1) * t
                    y_left = y_left1 + (y_left2 - y_left1) * t

                x_right1, y_right1 = x_mid, y_mid
                x_right2, y_right2 = 2*x_L - x_ap, 2*y_L -y_ap
                for label in Right_labels_rest:
                    x, y = label
                    cos_theta_1 = get_cos_theta(x_right1, y_right1, x_ap, y_ap, x_h, y_h)
                    cos_theta_2 = get_cos_theta(x_right2, y_right2, x_ap, y_ap, x_h, y_h)
                    cos_theta_label = get_cos_theta(x, y, x_ap, y_ap, x_h, y_h)

                    x_right1 = x * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + x_right1 * (1 - (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))
                    y_right1 = y * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + y_right1 * (1 - (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))

                    x_right2 = x * (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2) + x_right2 * (1 - (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2))
                    y_right2 = y * (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2) + y_right2 * (1 - (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2))

                checker2 = (x_right2 == x_right1) * (y_right2 == y_right1)
                if checker2 == 1:
                    x_right = x_right1
                    y_right = y_right1
                else:
                    t = np.dot([x_h - x_right1, y_h - y_right1], [x_ap - x_h, y_ap - y_h]) / np.dot([x_right2 - x_right1, y_right2 - y_right1], [x_ap - x_h, y_ap - y_h])
                    x_right = x_right1 + (x_right2 - x_right1) * t
                    y_right = y_right1 + (y_right2 - y_right1) * t

                distance_left_2_middle = math.sqrt( (x_left - x_h)**2 + (y_left - y_h)**2 )
                distance_right_2_middle = math.sqrt( (x_right - x_h)**2 + (y_right - y_h)**2 )

                LV_volume = LV_volume + math.pi * height * (distance_right_2_middle**2 - distance_left_2_middle**2)

    return LV_volume
    
def get_cos_theta(p1x, p1y, p2x, p2y, p3x, p3y):
    """Get the value of cosine of the angle consisting of three points

    Args:
      p1x, p1y, p2x, p2y, p3x, p3y: from a point of (p1x, p1y), a point of (p2x, p2y), and a point of (p3x, p3y)

    Return:
      the value of cosine of the angle between the line of point 1 to point 3 and the line of point 2 to point 3.
    """

    return np.dot([p1x - p3x, p1y - p3y], [p2x - p3x, p2y - p3y]) / ( math.sqrt( (p1x - p3x)**2 + (p1y - p3y)**2 ) * math.sqrt( (p2x - p3x)**2 + (p2y - p3y)**2 ) )

def get_segmented_line(base_vertices, N):
    """Get the value of cosine of the angle consisting of three points

    Args:
      base_vertices: the output of get_corrected_base_vertices
      N: 20 in typical, the number of discretized disks used to approximate the LV volume by integration 

    Returns:
      several coordinates for the implementation of Modified Simpson's Method
    """

    x_ap, y_ap = base_vertices[0]
    x_mid = ( base_vertices[1][0] + base_vertices[2][0] ) / 2.0
    y_mid = ( base_vertices[1][1] + base_vertices[2][1] ) / 2.0

    b_1x, b_1y = base_vertices[1]
    b_2x, b_2y = base_vertices[2]

    t_b1 = np.dot([x_mid - x_ap, y_mid - y_ap], [b_1x - x_ap, b_1y - y_ap]) / np.dot([x_mid - x_ap, y_mid - y_ap], [x_mid - x_ap, y_mid - y_ap])
    t_b2 = np.dot([x_mid - x_ap, y_mid - y_ap], [b_2x - x_ap, b_2y - y_ap]) / np.dot([x_mid - x_ap, y_mid - y_ap], [x_mid - x_ap, y_mid - y_ap])

    t_L = t_b1 * (t_b1 >= t_b2) + t_b2 * (t_b1 < t_b2)
    x_L = x_ap + (x_mid - x_ap) * t_L # outside point along the middle line
    y_L = y_ap + (y_mid - y_ap) * t_L # outside point along the middle line

    t_l = t_b1 * (t_b1 < t_b2) + t_b2 * (t_b1 >= t_b2)
    x_l = x_ap + (x_mid - x_ap) * t_l # inside point along the middle line
    y_l = y_ap + (y_mid - y_ap) * t_l # inside point along the middle line

    list_segmented_N_x = list(np.linspace(x_ap, x_L, N+1))
    list_segmented_N_y = list(np.linspace(y_ap, y_L, N+1))
    
    height = math.sqrt( (x_ap - x_L)**2 + (y_ap - y_L)**2 ) / N

    return list_segmented_N_x, list_segmented_N_y, x_l, y_l, x_L, y_L, height

def get_distinguish_parameters(base_vertices):
    """Get the parameters for the implementation of Modified Simpsons' Method

    Args:
      base_vertices: the output of get_corrected_base_vertices

    Returns:
      several parameters for the implementation of Modified Simpson's Method
    """

    x_ap, y_ap = base_vertices[0]

    b_1x, b_1y = base_vertices[1]
    b_2x, b_2y = base_vertices[2]

    K_numerator = (b_1x + b_2x - 2*x_ap) 
    K_denominator = (b_1y + b_2y - 2*y_ap) 

    K = K_numerator / K_denominator
    C = -K*y_ap + x_ap
    
    K_numerator_b1 = (b_1x - x_ap)
    K_denominator_b1 = (b_1y - y_ap)

    K_b1 = K_numerator_b1 / K_denominator_b1
    C_b1 = -K_b1*y_ap + x_ap

    K_numerator_b2 = (b_2x - x_ap)
    K_denominator_b2 = (b_2y - y_ap)

    K_b2 = K_numerator_b2 / K_denominator_b2
    C_b2 = -K_b2*y_ap + x_ap

    return K, C, K_b1, C_b1, K_b2, C_b2 

def get_sides_labels(base_vertices, labels_list):
    """Get the sides labels for finding the correct disks used to the integration of LV volume

    Args:
      base_vertices: the output of get_corrected_base_vertices
      labels_list: the outpout of get_labels

    Returns:
      the sides labels for finding the correct disks used to the integration of LV volume
    """

    K, C, K_b1, C_b1, K_b2, C_b2  = get_distinguish_parameters(base_vertices)
    x_ap, y_ap = base_vertices[0]
    x_mid = ( base_vertices[1][0] + base_vertices[2][0] ) / 2.0
    y_mid = ( base_vertices[1][1] + base_vertices[2][1] ) / 2.0

    distance_m_2_apex_cordis = math.sqrt( (x_mid - x_ap)**2 + (y_mid - y_ap)**2 )

    Left_labels_for_sort = []
    Right_labels_for_sort = []
    for label in labels_list:
        x_label, y_label = label
        if x_label < (K*y_label + C):
            distance_left_2_apex_cordis = math.sqrt( (x_label - x_ap)**2 + (y_label - y_ap)**2 )
            Left_labels_for_sort.append((label, distance_left_2_apex_cordis))
        elif x_label > (K*y_label + C):
            distance_right_2_apex_cordis = math.sqrt( (x_label - x_ap)**2 + (y_label - y_ap)**2 )
            Right_labels_for_sort.append((label, distance_right_2_apex_cordis))
    
    Left_labels_sorted = sorted(Left_labels_for_sort, key = lambda point:point[1])
    Right_labels_sorted = sorted(Right_labels_for_sort, key = lambda point:point[1])

    count_the_rest_left_labels = 0
    for element in range(0, len(Left_labels_sorted)):
        label, distance_left_2_apex_cordis = Left_labels_sorted[element]
        if distance_left_2_apex_cordis >= distance_m_2_apex_cordis:
            count_the_rest_left_labels += 1

    count_the_rest_right_labels = 0
    for element in range(0, len(Right_labels_sorted)):
        label, distance_right_2_apex_cordis = Right_labels_sorted[element]
        if distance_right_2_apex_cordis >= distance_m_2_apex_cordis:
            count_the_rest_right_labels += 1


    Left_labels_b1 = []
    Right_labels_b1 = []
    for num in range(0, len(Left_labels_sorted)):
        label, dist = Left_labels_sorted[num]
        x_label, y_label = label
        if x_label < (K_b1*y_label + C_b1):
            Left_labels_b1.append(label)
        elif x_label > (K_b1*y_label + C_b1):
            Right_labels_b1.append(label)

    Left_labels_by_middle = Left_labels_b1 + [base_vertices[1]] + Right_labels_b1

    Left_labels_b2 = []
    Right_labels_b2 = []
    for num in range(0, len(Right_labels_sorted)):
        label, dist = Right_labels_sorted[num]
        x_label, y_label = label
        if x_label < (K_b2*y_label + C_b2):
            Left_labels_b2.append(label)
        elif x_label > (K_b2*y_label + C_b2):
            Right_labels_b2.append(label)

    Right_labels_by_middle = Right_labels_b2 + [base_vertices[2]] + Left_labels_b2

    return Left_labels_by_middle, Right_labels_by_middle, Left_labels_b1, Right_labels_b1, Right_labels_b2, Left_labels_b2, count_the_rest_left_labels, count_the_rest_right_labels

def draw_on_picture(filename_json_dir, filename_png_dir):

    #filename_json_dir = 'H65138_20181210110344-051.json' 
    #filename_png_dir = 'H65138_20181210110344-051.png'
    labels_list = get_labels(filename_json_dir)
    largest_triangle = find_the_largest_triangle(labels_list)
    base_vertices = check_the_bottom_2_points(labels_list, largest_triangle)

    list_segmented_N_x, list_segmented_N_y, x_l, y_l, x_L, y_L, height = get_segmented_line(base_vertices, N)
    Left_labels_by_middle, Right_labels_by_middle, Left_labels_b1, Right_labels_b1, Right_labels_b2, Left_labels_b2, count_the_rest_left_labels, count_the_rest_right_labels = get_sides_labels(base_vertices, labels_list)

    img = Image.open(filename_png_dir)
    draw = ImageDraw.Draw(img)

    draw_base_vertics_list = []
    for point in base_vertices:
        x, y = point
        draw_base_vertics_list.append((x, y))
    draw_base_vertics_list.append(draw_base_vertics_list[0])

    draw.line(draw_base_vertics_list, fill = (255, 0, 0), width = 1)
    
    draw_labels_list = []
    for label in labels_list:
        x, y = label
        draw_labels_list.append((x, y))
    draw_labels_list.append(draw_labels_list[0])

    line_color = (0, 0, 255)

    draw.line(draw_labels_list, fill = line_color, width = 1)

    draw_middle_lint_list = list(zip(list_segmented_N_x, list_segmented_N_y))
    
    line_color = (0, 255, 0)

    draw.line(draw_middle_lint_list, fill = line_color, width = 1)
    #img.show()

    K, C, K_b1, C_b1, K_b2, C_b2 = get_distinguish_parameters(base_vertices)

    x_ap, y_ap = base_vertices[0]
    x_mid = ( base_vertices[1][0] + base_vertices[2][0] ) / 2.0
    y_mid = ( base_vertices[1][1] + base_vertices[2][1] ) / 2.0

    for segment in range(0, len(list_segmented_N_x)-1):
        x_h = list_segmented_N_x[segment + 1]
        y_h = list_segmented_N_y[segment + 1]
        
        distance_h_2_apex_cordis = math.sqrt( (x_h - x_ap)**2 + (y_h - y_ap)**2 )
        distance_l_2_apex_cordis = math.sqrt( (x_l - x_ap)**2 + (y_l - y_ap)**2 )
        distance_m_2_apex_cordis = math.sqrt( (x_mid - x_ap)**2 + (y_mid - y_ap)**2 )

        if distance_h_2_apex_cordis <= distance_m_2_apex_cordis:
            x_left1, y_left1 = x_ap, y_ap
            x_left2, y_left2 = x_mid, y_mid
            for label in Left_labels_by_middle:
                x, y = label
                cos_theta_1 = get_cos_theta(x_left1, y_left1, x_ap, y_ap, x_h, y_h)
                cos_theta_2 = get_cos_theta(x_left2, y_left2, x_ap, y_ap, x_h, y_h)
                cos_theta_label = get_cos_theta(x, y, x_ap, y_ap, x_h, y_h)

                x_left1 = x * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + x_left1 * (1- (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))
                y_left1 = y * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + y_left1 * (1- (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))

                x_left2 = x * (cos_theta_label < 0) * (cos_theta_label > cos_theta_2) + x_left2 * (1 - (cos_theta_label < 0) * (cos_theta_label > cos_theta_2))
                y_left2 = y * (cos_theta_label < 0) * (cos_theta_label > cos_theta_2) + y_left2 * (1 - (cos_theta_label < 0) * (cos_theta_label > cos_theta_2))
                
            checker1 = (x_left2 == x_left1) * (y_left2 == y_left1)
            if checker1 == 1:
                x_left = x_left1
                y_left = y_left1
            else:
                t = np.dot([x_h - x_left1, y_h - y_left1], [x_ap - x_h, y_ap - y_h]) / np.dot([x_left2 - x_left1, y_left2 - y_left1], [x_ap - x_h, y_ap - y_h])
                x_left = x_left1 + (x_left2 - x_left1) * t
                y_left = y_left1 + (y_left2 - y_left1) * t

            x_right1, y_right1 = x_ap, y_ap
            x_right2, y_right2 = x_mid, y_mid
            for label in Right_labels_by_middle:
                x, y = label
                cos_theta_1 = get_cos_theta(x_right1, y_right1, x_ap, y_ap, x_h, y_h)
                cos_theta_2 = get_cos_theta(x_right2, y_right2, x_ap, y_ap, x_h, y_h)
                cos_theta_label = get_cos_theta(x, y, x_ap, y_ap, x_h, y_h)

                x_right1 = x * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + x_right1 * (1 - (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))
                y_right1 = y * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + y_right1 * (1 - (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))

                x_right2 = x * (cos_theta_label < 0) * (cos_theta_label > cos_theta_2) + x_right2 * (1 - (cos_theta_label < 0) * (cos_theta_label > cos_theta_2))
                y_right2 = y * (cos_theta_label < 0) * (cos_theta_label > cos_theta_2) + y_right2 * (1 - (cos_theta_label < 0) * (cos_theta_label > cos_theta_2))
            
            checker2 = (x_right2 == x_right1) * (y_right2 == y_right1)
            if checker2 == 1:
                x_right = x_right1
                y_right = y_right1
            else:
                t = np.dot([x_h - x_right1, y_h - y_right1], [x_ap - x_h, y_ap - y_h]) / np.dot([x_right2 - x_right1, y_right2 - y_right1], [x_ap - x_h, y_ap - y_h])
                x_right = x_right1 + (x_right2 - x_right1) * t
                y_right = y_right1 + (y_right2 - y_right1) * t

            draw.line([(x_left, y_left), (x_right, y_right)], fill = line_color, width = 1)

        elif distance_h_2_apex_cordis > distance_m_2_apex_cordis:
            if count_the_rest_left_labels >= count_the_rest_right_labels:
                Left_labels_rest = []
                Right_labels_rest = []
                for label in Left_labels_by_middle:
                    x_label, y_label = label
                    if x_label < (K_b1*y_label + C_b1):
                        Left_labels_rest.append(label)
                    elif x_label > (K_b1*y_label + C_b1):
                        Right_labels_rest.append(label)
                                
                                
                Left_labels_rest =  Left_labels_rest + [base_vertices[1]]
                Right_labels_rest = Right_labels_rest + [base_vertices[1]]
                
                x_left1, y_left1 = x_mid, y_mid
                x_left2, y_left2 = 2*x_L - x_ap, 2*y_L -y_ap
                for label in Left_labels_rest:
                    x, y = label
                    cos_theta_1 = get_cos_theta(x_left1, y_left1, x_ap, y_ap, x_h, y_h)
                    cos_theta_2 = get_cos_theta(x_left2, y_left2, x_ap, y_ap, x_h, y_h)
                    cos_theta_label = get_cos_theta(x, y, x_ap, y_ap, x_h, y_h)

                    x_left1 = x * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + x_left1 * (1- (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))
                    y_left1 = y * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + y_left1 * (1- (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))

                    x_left2 = x * (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2) + x_left2 * (1 - (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2))
                    y_left2 = y * (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2) + y_left2 * (1 - (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2))
                
                checker1 = (x_left2 == x_left1) * (y_left2 == y_left1)
                if checker1 == 1:
                    x_left = x_left1
                    y_left = y_left1
                else:
                    t = np.dot([x_h - x_left1, y_h - y_left1], [x_ap - x_h, y_ap - y_h]) / np.dot([x_left2 - x_left1, y_left2 - y_left1], [x_ap - x_h, y_ap - y_h])
                    x_left = x_left1 + (x_left2 - x_left1) * t
                    y_left = y_left1 + (y_left2 - y_left1) * t

                x_right1, y_right1 = x_mid, y_mid
                x_right2, y_right2 = 2*x_L - x_ap, 2*y_L -y_ap
                for label in Right_labels_rest:
                    x, y = label
                    cos_theta_1 = get_cos_theta(x_right1, y_right1, x_ap, y_ap, x_h, y_h)
                    cos_theta_2 = get_cos_theta(x_right2, y_right2, x_ap, y_ap, x_h, y_h)
                    cos_theta_label = get_cos_theta(x, y, x_ap, y_ap, x_h, y_h)

                    cos_theta_1 = np.dot([x_right1 - x_h, y_right1 - y_h], [x_ap - x_h, y_ap - y_h]) / ( math.sqrt( (x_right1 - x_h)**2 + (y_right1 - y_h)**2 ) * math.sqrt( (x_ap - x_h)**2 + (y_ap - y_h)**2 ) )
                    cos_theta_2 = np.dot([x_right2 - x_h, y_right2 - y_h], [x_ap - x_h, y_ap - y_h]) / ( math.sqrt( (x_right2 - x_h)**2 + (y_right2 - y_h)**2 ) * math.sqrt( (x_ap - x_h)**2 + (y_ap - y_h)**2 ) )
                    cos_theta_label = np.dot([x - x_h, y - y_h], [x_ap - x_h, y_ap - y_h]) / ( math.sqrt( (x - x_h)**2 + (y - y_h)**2 ) * math.sqrt( (x_ap - x_h)**2 + (y_ap - y_h)**2 ) )

                    x_right1 = x * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + x_right1 * (1 - (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))
                    y_right1 = y * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + y_right1 * (1 - (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))

                    x_right2 = x * (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2) + x_right2 * (1 - (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2))
                    y_right2 = y * (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2) + y_right2 * (1 - (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2))

                checker2 = (x_right2 == x_right1) * (y_right2 == y_right1)
                if checker2 == 1:
                    x_right = x_right1
                    y_right = y_right1
                else:
                    t = np.dot([x_h - x_right1, y_h - y_right1], [x_ap - x_h, y_ap - y_h]) / np.dot([x_right2 - x_right1, y_right2 - y_right1], [x_ap - x_h, y_ap - y_h])
                    x_right = x_right1 + (x_right2 - x_right1) * t
                    y_right = y_right1 + (y_right2 - y_right1) * t

                draw.line([(x_left, y_left), (x_right, y_right)], fill = line_color, width = 1)

            else:
                Left_labels_rest = []
                Right_labels_rest = []
                for label in Right_labels_by_middle:
                    x_label, y_label = label
                    if x_label < (K_b2*y_label + C_b2):
                        Left_labels_rest.append(label)
                    elif x_label > (K_b2*y_label + C_b2):
                        Right_labels_rest.append(label)

                Left_labels_rest = Left_labels_rest + [base_vertices[2]]
                Right_labels_rest = Right_labels_rest + [base_vertices[2]]

                x_left1, y_left1 = x_mid, y_mid
                x_left2, y_left2 = 2*x_L - x_ap, 2*y_L -y_ap
                for label in Left_labels_rest:
                    x, y = label
                    cos_theta_1 = get_cos_theta(x_left1, y_left1, x_ap, y_ap, x_h, y_h)
                    cos_theta_2 = get_cos_theta(x_left2, y_left2, x_ap, y_ap, x_h, y_h)
                    cos_theta_label = get_cos_theta(x, y, x_ap, y_ap, x_h, y_h)

                    x_left1 = x * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + x_left1 * (1- (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))
                    y_left1 = y * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + y_left1 * (1- (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))

                    x_left2 = x * (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2) + x_left2 * (1 - (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2))
                    y_left2 = y * (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2) + y_left2 * (1 - (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2))
                
                checker1 = (x_left2 == x_left1) * (y_left2 == y_left1)
                if checker1 == 1:
                    x_left = x_left1
                    y_left = y_left1
                else:
                    t = np.dot([x_h - x_left1, y_h - y_left1], [x_ap - x_h, y_ap - y_h]) / np.dot([x_left2 - x_left1, y_left2 - y_left1], [x_ap - x_h, y_ap - y_h])
                    x_left = x_left1 + (x_left2 - x_left1) * t
                    y_left = y_left1 + (y_left2 - y_left1) * t

                x_right1, y_right1 = x_mid, y_mid
                x_right2, y_right2 = 2*x_L - x_ap, 2*y_L -y_ap
                for label in Right_labels_rest:
                    x, y = label
                    cos_theta_1 = get_cos_theta(x_right1, y_right1, x_ap, y_ap, x_h, y_h)
                    cos_theta_2 = get_cos_theta(x_right2, y_right2, x_ap, y_ap, x_h, y_h)
                    cos_theta_label = get_cos_theta(x, y, x_ap, y_ap, x_h, y_h)

                    x_right1 = x * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + x_right1 * (1 - (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))
                    y_right1 = y * (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1) + y_right1 * (1 - (cos_theta_label >= 0) * (cos_theta_label < cos_theta_1))

                    x_right2 = x * (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2) + x_right2 * (1 - (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2))
                    y_right2 = y * (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2) + y_right2 * (1 - (cos_theta_label <= 0) * (cos_theta_label > cos_theta_2))

                checker2 = (x_right2 == x_right1) * (y_right2 == y_right1)
                if checker2 == 1:
                    x_right = x_right1
                    y_right = y_right1
                else:
                    t = np.dot([x_h - x_right1, y_h - y_right1], [x_ap - x_h, y_ap - y_h]) / np.dot([x_right2 - x_right1, y_right2 - y_right1], [x_ap - x_h, y_ap - y_h])
                    x_right = x_right1 + (x_right2 - x_right1) * t
                    y_right = y_right1 + (y_right2 - y_right1) * t

                draw.line([(x_left, y_left), (x_right, y_right)], fill = line_color, width = 1)       

    img.show()

def nothing():
    import matplotlib.pyplot as plt 
    filenames = []
    for filename in glob.glob('*.json'):
        filenames.append(filename)

    filenames.sort()
    
    LV_volumes = [] 
    for filename_json_dir in filenames:
        value =  Modified_Simpson_calculator(filename_json_dir) 
        if value == 'nan':
            LV_volumes.append(0) 
        else: 
            LV_volumes.append(value) 

    EDV = max(LV_volumes)
    ESV = min(LV_volumes)
    EF = (EDV - ESV) * 100 / EDV
    print('EF = {}'.format(EF))

    x = list(range(0, len(LV_volumes))) 
    plt.plot(x, LV_volumes)
    plt.xlabel('Imgae frame')
    plt.ylabel('LV volume (unitless)')
    plt.title('The variation of LV volume ( EF = {} % )'.format(EF))
    plt.show()

    #for filename_json_dir in filenames: 
    #    print(filename_json_dir + ' LV_volume = {}'.format(Modified_Simpson_calculator(filename_json_dir))) 