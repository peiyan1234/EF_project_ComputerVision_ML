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

def calculator(filename_json):
    """
    This function calculate the volumn of a left ventricle.
    """
    labels = get_labels(filename_json)


def get_labels(filename_json):
    """
    This function outputs the coordinations of labels on an ultrasound image
    """
