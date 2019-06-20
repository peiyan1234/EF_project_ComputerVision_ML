from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
from PIL import Image

import numpy as np
import tensorflow as tf

import Unet

IMAGE_SIZE = (800, 600)
height, width = IMAGE_SIZE

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/alvinli/Desktop/EF/dataset/EF-training-Pool/eval',"""Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',"""Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/alvinli/Desktop/EF/dataset/EF-training-Pool/train',"""Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 1,"""How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 121,"""Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,"""Whether to run eval only once.""")

def show():
    """
    show the predicted mask for the segmentation purpose. 
    """
    
    "do something"
        

def main(argv=None):
    show()

if __name__ == '__main__':
  tf.app.run()
