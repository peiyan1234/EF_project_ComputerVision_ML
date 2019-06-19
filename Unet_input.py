from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import math
import glob
import json
import os
import sys
import shutil

from six.moves import xrange  
import tensorflow as tf
import numpy as np

IMAGE_SIZE = (800, 600)
height, width = IMAGE_SIZE

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 485
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 121

dir_training_pool = '/home/alvinli/Desktop/EF/dataset/EF-training-Pool'
dir_datasheet = os.path.join(dir_training_pool,'datasheet.json')

def distorted_inputs(dir_data, batch_size):
    """Generate distorted input for Unet training. 

    Args:
        dir_data: path to the data
        batch_size: number of images per batch
    
    Returns:
        images: Images, 4D tensor of [batch_size, width, height, 1] size.
        labels: Labels, 4D tensor of [batch_size, width, height, 1] size.

    """
    label_images = []
    input_images = []
    for name in glob.glob(os.path.join(dir_data, '*_ROI_mask.png')):
        label_images.append(name)
        input_images.append(name.replace("_ROI_mask.png",".png"))

    queue = input_images
    with tf.name_scope('data_augmentation'):
        read_input = read_data(queue, label_images)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)
        reshaped_label = tf.cast(read_input.uint8label, tf.float32)

        float_image = tf.image.per_image_standardization(reshaped_image)
        
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

        print ('Filling queue with %d EF images before starting to train. '
               'This will take a few minutes.' % min_queue_examples)

    return _generate_image_and_label_batch(float_image, reshaped_label,
                                          min_queue_examples, batch_size,
                                          shuffle=True)
        

def read_data(queue, label_images):
    """Reads and Parses examples

    Notice: If you want N-way read parallelism, call this function N times.
            This will give you N independent Readers reading different files
            and positions within those files, which will give better mixing of
            examples.

    Args:
        queue: a list of filenames.
        label_images: a list of label-image filenames 

    Returns:
        An object representing a single example, with the following fields:
            height: number of rows in the result
            width:  number of columns in the result
            depth:  number of color channels in the result
            uint8image: a [height, width, depth] uint8 Tensor with the image data
            uint8label:  a [height, width, depth] uint8 Tensor with the label information

    """
    class dataRecord(object):
        pass
    result = dataRecord()

    result.width, result.height = IMAGE_SIZE
    result.depth = 1

    tf_filenames = tf.convert_to_tensor(queue, dtype=tf.string)
    tf_labels = tf.convert_to_tensor(label_images, dtype=tf.string)

    print('the shape of tf_filenames: {}'.format(tf_filenames))
    print('the shape of tf_labels: {}'.format(tf_labels))

    input_queue = tf.train.slice_input_producer([tf_filenames, tf_labels], shuffle=False)

    image_contents = tf.read_file(input_queue[0])
    result.uint8image = tf.image.decode_png(image_contents,channels=1, dtype=tf.dtypes.uint8)

    label_contents = tf.read_file(input_queue[1])
    result.uint8label = tf.image.decode_png(label_contents,channels=1, dtype=tf.dtypes.uint8)

    return result

def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    """generate a queued batch of images and labels

    Args:
        image: 3D Tensor of [height, width, 1] of type.float32
        label: 3D Tensor of [height, width, 1] of type.float32
        min_queue_examples: int32, minimum number of samples to retain in the queue that provides of batches of examples
        batch_size: number of images per batch
        shuffle: boolean indicating whether to use a shuffling queue

    Returns:
        images: Images, 4D Tensor of [batch_size, height, width, 1] size
        labels: Labels, 4D Tensor of [batch_size, height, width, 1] size

    """
    num_preprocess_threads = 4
    if shuffle:
        images, labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples,
            shapes = [(width, height, 1), (width, height, 1)])
    else:
        images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            shapes = [(width, height, 1), (width, height, 1)])

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, labels

def inputs(eval_data, dir_data, batch_size):
    """Construct input for evaluation using the Reader ops.

    Args:
        eval_data: bool, indicating if one should use the train or eval data set.
        dir_data: path to the data.
        batch_size: number of images per batch

    Returns:
        images: Images, 4D tensor of [batch_size, width, height, 1] size.
        labels: Labels, 4D tensor of [batch_size, width, height, 1] size.
    """
    label_images = []
    input_images = []
    for name in glob.glob(os.path.join(dir_data, '*_ROI_mask.png')):
        label_images.append(name)
        input_images.append(name.replace("_ROI_mask.png",".png"))

    queue = input_images
    with tf.name_scope('input'):
        read_input = read_data(queue, label_images)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)
        reshaped_label = tf.cast(read_input.uint8label, tf.float32)

        float_image = tf.image.per_image_standardization(reshaped_image)
        
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

        print ('Filling queue with %d EF images before starting to train. '
               'This will take a few minutes.' % min_queue_examples)

    return _generate_image_and_label_batch(float_image, reshaped_label,
                                          min_queue_examples, batch_size,
                                          shuffle=True)