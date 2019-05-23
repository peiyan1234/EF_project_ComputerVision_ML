import json
import os
import glob
import math
import numpy as np
from itertools import combinations
from PIL import Image, ImageDraw

"""
Boundary fill algorithm bases on 8-connected pixels.
"""

def bresenham_line_algorithm(x1, y1, x2, y2):
    """draw pixels between (x1, y1) and (x2, y2) to form a straight line 

    Arg:
      x1, y1: the coordinate of point1
      x2, y2: the coordinate of point2

    Return:
      points: a list of points, e.g. [[px1, py1], [px2, py2], ...]
    """

    if abs(y2 - y1) < abs(x2 - x1):
        if x1 > x2:
            return plotLineLow(x2, y2, x1, y1)
        else:
            return plotLineLow(x1, y1, x2, y2)
    else:
        if y1 > y2:
            return plotLineHigh(x2, y2, x1, y1)
        else:
            return plotLineHigh(x1, y1, x2, y2)

def plotLineLow(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    yi = 1 * (dy >= 0) + -1 * (dy < 0)
    dy = dy * (dy >= 0) + -1 * dy * (dy < 0)
    D = 2 * dy - dx
    y = y1

    points = []
    for x in range(x1, x2 + 1):
        points.append([x, y])
        y += yi * (D > 0)
        D += 2 * dy - 2 * dx * (D > 0)
            
    return points

def plotLineHigh(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    xi = 1 * (dx >= 0) + -1 * (dx < 0)
    dx = dx * (dx >= 0) + -1 * dx * (dx < 0)
    D = 2 * dx - dy
    x = x1

    points = []
    for y in range(y1, y2 + 1):
        points.append([x, y])
        x += xi * (D > 0)
        D += 2 * dx - 2 * dy * (D > 0)

    return points


def BoundaryFill8(x, y, fill_color, boundary_color):
    """Fill the region inside a closed contour

    Arg:
      x, y: the seed's location which should be inside the region of a closed contour
      fill_color: gray level from 0 to 255
      boundary_color: gray level from 0 to 255

    Return:
      N/A
    """

""" Reference code
void boundaryFill8(int x, int y, int fill_color,int boundary_color)
{
    if(getpixel(x, y) != boundary_color &&
       getpixel(x, y) != fill_color)
    {
        putpixel(x, y, fill_color);
        boundaryFill8(x + 1, y, fill_color, boundary_color);
        boundaryFill8(x, y + 1, fill_color, boundary_color);
        boundaryFill8(x - 1, y, fill_color, boundary_color);
        boundaryFill8(x, y - 1, fill_color, boundary_color);
        boundaryFill8(x - 1, y - 1, fill_color, boundary_color);
        boundaryFill8(x - 1, y + 1, fill_color, boundary_color);
        boundaryFill8(x + 1, y - 1, fill_color, boundary_color);
        boundaryFill8(x + 1, y + 1, fill_color, boundary_color);
    }
}

"""

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

#labels_list = get_labels(filename_json_dir)

x1, y1 = labels_list[0]
x2, y2 = labels_list[1]
bresenham_line_algorithm(x1, y1, x2, y2)

contour_list = []
N = len(labels_list)
for n in range(0, N - 1):
    x1, y1 = labels_list[n]
    x2, y2 = labels_list[n+1]
    points = bresenham_line_algorithm(x1, y1, x2, y2)
    if points[0] == labels_list[n+1]:
        points.reverse()
    M = len(points)
    for m in range(0, M - 1):
        contour_list.append(points[m])

x1, y1 = labels_list[N - 1]
x2, y2 = labels_list[0]
points = bresenham_line_algorithm(x1, y1, x2, y2)
if points[0] == labels_list[0]:
    points.reverse()
M = len(points)
for m in range(0, M - 1):
    contour_list.append(points[m])

[label1, label2, label3] = find_the_largest_triangle(labels_list)
seed_x = (label1[0] + label2[0] + label3[0]) / 3
seed_y = (label1[1] + label2[0] + label3[0]) / 3

img = Image.open(filename_png_dir)
img = np.asarray(img)
img_width, img_height, img_depth = img.shape
ROI_mask = np.zeros(img_width, img_height)
