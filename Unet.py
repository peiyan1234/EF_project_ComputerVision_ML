from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import Unet_input

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 1, 'Number of images to process in a batch')
tf.app.flags.DEFINE_string('dir_data', '/home/alvinli/Desktop/EF/dataset/EF-training-Pool', 'Path to the EF data directory')
tf.app.flags.DEFINE_boolean('use_fp16', False, 'Train the model using fp16')

IMAGE_SIZE = Unet_input.IMAGE_SIZE

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = Unet_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = Unet_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 100
LEARNING_RATE_DECAY_FACTOR = 0.12
INITIAL_LEARNING_RATE = 0.12

"""
If a model is trained with multiple GPUs, prefix all op names with tower_name
to differentiate the operations. Note that this prefix is removed from the names
of the summaries when visualizing a model.
"""
TOWER_NAME = 'tower'

def _activation_summary(x):
    """generate summaries for activations

    1. Creates a summary that provides a histogram of activations.
    2. Creates a summary that measures the sparsity of activations.

    Args:
        x: Tensor

    Returns:

    """
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
    """Generate a Variable stored on CPU memory

    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

    Returns:
        Variable Tensor

    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    
    return tf.get_variable(name, shape, initializer=initializer, dtype=dtype)

def _variable_with_weight_decay(name, shape, stddev, wd):
    """Generate an initialized Variable with weight decay

    Args: 
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2 loss weight decay multiplied by this float. If None, weight decay is not added for this Variable.

    Returns:
        Variable Tensor

    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name = 'weight_loss')
        tf.add_to_collection('losses', weight_decay)
    
    return var

def distorted_inputs():
    """Generate distorted input for model training using the Reader ops

    Args:

    Returns:
        images: Images, 4D tensor of [batch_size, width, height, 1] size
        labels: Labels, 4D tensor of [batch_size, width, height, 1] size

    Raises:
        ValueError: If no dir_data

    """
    if not FLAGS.dir_data:
        raise ValueError('Please supply a dir_data')

    dir_data = os.path.join(FLAGS.dir_data, 'batches', 'train_batch')
    images, labels = Unet_input.distorted_inputs(dir_data=dir_data, batch_size=FLAGS.batch_size)

    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)

    return images, labels

def inputs(eval_data):
    """Construct input for evaluation using the Reader ops.

    Args:
        eval_data: bool, indicating if one should use the train or eval data set.

    Returns:
        images: Images, 4D tensor of [batch_size, width, height, 1] size
        labels: Labels, 4D tensor of [batch_size, width, height, 1] size

    Raises:
        ValueError: If no data_dir
    """
    if not FLAGS.dir_data:
        raise ValueError('Please supply a data_dir')
    dir_data = os.path.join(FLAGS.dir_data, 'batches', 'test_batch')
    images, labels = Unet_input.inputs(eval_data=eval_data,
                                        dir_data=dir_data,
                                        batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    
    return images, labels

def inference(images):
    """Build the Unet model.
    
    All variables are instantiated by tf.get_variable() instead of tf.Variable() in order to 
    share variables across multiple GPU training runs. 

    If we only run this model on a single GPU, we could simplify this function by replacing all
    instances of tf.get_variable() with tf.Variable()
    
    Args:
        images: Images returned from distorted_inputs()
    
    Returns:
        conv10

    """

    ##########################
    #Contraction Path Encoder#
    ##########################


    # 2@ConvLayers, 16@ 3x3 filters, Padding = 'same'
    # Max pool 2x2 filter, strides = 2
    # dropout rate = 0.1

    with tf.variable_scope('1st_block_conv2d') as scope:
        kernel1 = _variable_with_weight_decay('weights1', shape = [3, 3, 1, 16], stddev = 5e-2, wd = None)
        conv = tf.nn.conv2d(images, kernel1, [1, 1, 1, 1], padding='SAME')
        biases1 = _variable_on_cpu('biases1', [16], tf.constant_initializer(0.1))
        pre_activation1 = tf.nn.bias_add(conv, biases1)
        conv = tf.nn.relu(pre_activation1, name=scope.name)

        kernel2 = _variable_with_weight_decay('weights2', shape = [3, 3, 16, 16], stddev = 5e-2, wd = None)
        conv1 = tf.nn.conv2d(conv, kernel2, [1, 1, 1, 1], padding='SAME')
        biases2 = _variable_on_cpu('biases2', [16], tf.constant_initializer(0.1))
        pre_activation2 = tf.nn.bias_add(conv1, biases2)
        conv1 = tf.nn.relu(pre_activation2, name=scope.name)

        _activation_summary(conv1)

    pool_1st = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = '1st_pool')
    pool_1st = tf.nn.dropout(pool_1st, rate = 0.1)

    norm_1st = tf.nn.lrn(pool_1st, 1, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='1st_norm')

    # 2@ConvLayers, 32@ 3x3 filters, Padding = 'same'
    # Max pool 2x2 filter, strides = 2
    # dropout rate = 0.1

    with tf.variable_scope('2nd_block_conv2d') as scope:
        kernel1 = _variable_with_weight_decay('weights1', shape = [3, 3, 16, 32], stddev = 5e-2, wd = None)
        conv = tf.nn.conv2d(norm_1st, kernel1, [1, 1, 1, 1], padding='SAME')
        biases1 = _variable_on_cpu('biases1', [32], tf.constant_initializer(0.1))
        pre_activation1 = tf.nn.bias_add(conv, biases1)
        conv = tf.nn.relu(pre_activation1, name=scope.name)

        kernel2 = _variable_with_weight_decay('weights2', shape = [3, 3, 32, 32], stddev = 5e-2, wd = None)
        conv2 = tf.nn.conv2d(conv, kernel2, [1, 1, 1, 1], padding='SAME')
        biases2 = _variable_on_cpu('biases2', [32], tf.constant_initializer(0.1))
        pre_activation2 = tf.nn.bias_add(conv2, biases2)
        conv2 = tf.nn.relu(pre_activation2, name=scope.name)

        _activation_summary(conv2)

    pool_2nd = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = '2nd_pool')
    pool_2nd = tf.nn.dropout(pool_2nd, rate = 0.1)

    norm_2nd = tf.nn.lrn(pool_2nd, 1, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='2nd_norm')

    # 2@ConvLayers, 64@ 3x3 filters, Padding = 'same'
    # Max pool 2x2 filter, strides = 2
    # dropout rate = 0.1

    with tf.variable_scope('3rd_block_conv2d') as scope:
        kernel1 = _variable_with_weight_decay('weights1', shape = [3, 3, 32, 64], stddev = 5e-2, wd = None)
        conv = tf.nn.conv2d(norm_2nd, kernel1, [1, 1, 1, 1], padding='SAME')
        biases1 = _variable_on_cpu('biases1', [64], tf.constant_initializer(0.1))
        pre_activation1 = tf.nn.bias_add(conv, biases1)
        conv = tf.nn.relu(pre_activation1, name=scope.name)

        kernel2 = _variable_with_weight_decay('weights2', shape = [3, 3, 64, 64], stddev = 5e-2, wd = None)
        conv3 = tf.nn.conv2d(conv, kernel2, [1, 1, 1, 1], padding='SAME')
        biases2 = _variable_on_cpu('biases2', [64], tf.constant_initializer(0.1))
        pre_activation2 = tf.nn.bias_add(conv3, biases2)
        conv3 = tf.nn.relu(pre_activation2, name=scope.name)

        _activation_summary(conv3)

    pool_3rd = tf.nn.max_pool(conv3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = '3rd_pool')
    pool_3rd = tf.nn.dropout(pool_3rd, rate = 0.1)

    norm_3rd = tf.nn.lrn(pool_3rd, 1, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='3rd_norm')

    # 2@ConvLayers, 128@ 3x3 filters, Padding = 'same'
    # Max pool 5x5 filter, strides = 5
    # dropout rate = 0.1

    with tf.variable_scope('4th_block_conv2d') as scope:
        kernel1 = _variable_with_weight_decay('weights1', shape = [3, 3, 64, 128], stddev = 5e-2, wd = None)
        conv = tf.nn.conv2d(norm_3rd, kernel1, [1, 1, 1, 1], padding='SAME')
        biases1 = _variable_on_cpu('biases1', [128], tf.constant_initializer(0.1))
        pre_activation1 = tf.nn.bias_add(conv, biases1)
        conv = tf.nn.relu(pre_activation1, name=scope.name)

        kernel2 = _variable_with_weight_decay('weights2', shape = [3, 3, 128, 128], stddev = 5e-2, wd = None)
        conv4 = tf.nn.conv2d(conv, kernel2, [1, 1, 1, 1], padding='SAME')
        biases2 = _variable_on_cpu('biases2', [128], tf.constant_initializer(0.1))
        pre_activation2 = tf.nn.bias_add(conv4, biases2)
        conv4 = tf.nn.relu(pre_activation2, name=scope.name)

        _activation_summary(conv4)

    pool_4th = tf.nn.max_pool(conv4, ksize = [1, 5, 5, 1], strides = [1, 5, 5, 1], padding = 'SAME', name = '4th_pool')
    pool_4th = tf.nn.dropout(pool_4th, rate = 0.1)

    norm_4th = tf.nn.lrn(pool_4th, 1, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='4th_norm')

    # 2@ConvLayers, 256@ 3x3 filters, Padding = 'same'

    with tf.variable_scope('5th_block_conv2d') as scope:
        kernel1 = _variable_with_weight_decay('weights1', shape = [3, 3, 128, 256], stddev = 5e-2, wd = None)
        conv = tf.nn.conv2d(norm_4th, kernel1, [1, 1, 1, 1], padding='SAME')
        biases1 = _variable_on_cpu('biases1', [256], tf.constant_initializer(0.1))
        pre_activation1 = tf.nn.bias_add(conv, biases1)
        conv = tf.nn.relu(pre_activation1, name=scope.name)

        kernel2 = _variable_with_weight_decay('weights2', shape = [3, 3, 256, 256], stddev = 5e-2, wd = None)
        conv5 = tf.nn.conv2d(conv, kernel2, [1, 1, 1, 1], padding='SAME')
        biases2 = _variable_on_cpu('biases2', [256], tf.constant_initializer(0.1))
        pre_activation2 = tf.nn.bias_add(conv5, biases2)
        conv5 = tf.nn.relu(pre_activation2, name=scope.name)

        _activation_summary(conv5)

    norm_5th = tf.nn.lrn(conv5, 1, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='5th_norm')


    ########################
    #Expansion Path Decoder#
    ########################


    # conv2d_transpose
    # concatenate along the axis of channels
    
    with tf.variable_scope('6th_Unsample_Transposed_conv2d') as scope:
        kernel1 = _variable_with_weight_decay('weights1', shape = [5, 5, 128, 256], stddev = 5e-2, wd = None)
        upcon6 = tf.nn.conv2d_transpose(norm_5th, kernel1, output_shape = tf.shape(conv4), strides = [1, 5, 5, 1], padding='SAME')
        upcon6 = tf.concat([upcon6, conv4], 3)

    upcon6 = tf.nn.dropout(upcon6, rate = 0.1)

    # 2@ConvLayers, 128@ 3x3 filters, Padding = 'same'

    with tf.variable_scope('6th_block_conv2d') as scope:
        kernel1 = _variable_with_weight_decay('weights1', shape = [3, 3, 256, 128], stddev = 5e-2, wd = None)
        conv = tf.nn.conv2d(upcon6, kernel1, [1, 1, 1, 1], padding='SAME')
        biases1 = _variable_on_cpu('biases1', [128], tf.constant_initializer(0.1))
        pre_activation1 = tf.nn.bias_add(conv, biases1)
        conv = tf.nn.relu(pre_activation1, name=scope.name)

        kernel2 = _variable_with_weight_decay('weights2', shape = [3, 3, 128, 128], stddev = 5e-2, wd = None)
        conv6 = tf.nn.conv2d(conv, kernel2, [1, 1, 1, 1], padding='SAME')
        biases2 = _variable_on_cpu('biases2', [128], tf.constant_initializer(0.1))
        pre_activation2 = tf.nn.bias_add(conv6, biases2)
        conv6 = tf.nn.relu(pre_activation2, name=scope.name)

        _activation_summary(conv6)

    norm_6th = tf.nn.lrn(conv6, 1, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='6th_norm')

    # conv2d_transpose
    # concatenate along the axis of channels

    with tf.variable_scope('7th_Unsample_Transposed_conv2d') as scope:
        kernel1 = _variable_with_weight_decay('weights1', shape = [3, 3, 64, 128], stddev = 5e-2, wd = None)
        upcon7 = tf.nn.conv2d_transpose(norm_6th, kernel1, output_shape = tf.shape(conv3), strides = [1, 2, 2, 1], padding='SAME')
        upcon7 = tf.concat([upcon7, conv3], 3)

    upcon7 = tf.nn.dropout(upcon7, rate = 0.1)

    # 2@ConvLayers, 64@ 3x3 filters, Padding = 'same'

    with tf.variable_scope('7th_block_conv2d') as scope:
        kernel1 = _variable_with_weight_decay('weights1', shape = [3, 3, 128, 64], stddev = 5e-2, wd = None)
        conv = tf.nn.conv2d(upcon7, kernel1, [1, 1, 1, 1], padding='SAME')
        biases1 = _variable_on_cpu('biases1', [64], tf.constant_initializer(0.1))
        pre_activation1 = tf.nn.bias_add(conv, biases1)
        conv = tf.nn.relu(pre_activation1, name=scope.name)

        kernel2 = _variable_with_weight_decay('weights2', shape = [3, 3, 64, 64], stddev = 5e-2, wd = None)
        conv7 = tf.nn.conv2d(conv, kernel2, [1, 1, 1, 1], padding='SAME')
        biases2 = _variable_on_cpu('biases2', [64], tf.constant_initializer(0.1))
        pre_activation2 = tf.nn.bias_add(conv7, biases2)
        conv7 = tf.nn.relu(pre_activation2, name=scope.name)

        _activation_summary(conv7)

    norm_7th = tf.nn.lrn(conv7, 1, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='7th_norm')

    # conv2d_transpose
    # concatenate along the axis of channels

    with tf.variable_scope('8th_Unsample_Transposed_conv2d') as scope:
        kernel1 = _variable_with_weight_decay('weights1', shape = [3, 3, 32, 64], stddev = 5e-2, wd = None)
        upcon8 = tf.nn.conv2d_transpose(norm_7th, kernel1, output_shape = tf.shape(conv2), strides = [1, 2, 2, 1], padding='SAME')
        upcon8 = tf.concat([upcon8, conv2], 3)

    upcon8 = tf.nn.dropout(upcon8, rate = 0.1)

    # 2@ConvLayers, 32@ 3x3 filters, Padding = 'same'

    with tf.variable_scope('8th_block_conv2d') as scope:
        kernel1 = _variable_with_weight_decay('weights1', shape = [3, 3, 64, 32], stddev = 5e-2, wd = None)
        conv = tf.nn.conv2d(upcon8, kernel1, [1, 1, 1, 1], padding='SAME')
        biases1 = _variable_on_cpu('biases1', [32], tf.constant_initializer(0.1))
        pre_activation1 = tf.nn.bias_add(conv, biases1)
        conv = tf.nn.relu(pre_activation1, name=scope.name)

        kernel2 = _variable_with_weight_decay('weights2', shape = [3, 3, 32, 32], stddev = 5e-2, wd = None)
        conv8 = tf.nn.conv2d(conv, kernel2, [1, 1, 1, 1], padding='SAME')
        biases2 = _variable_on_cpu('biases2', [32], tf.constant_initializer(0.1))
        pre_activation2 = tf.nn.bias_add(conv8, biases2)
        conv8 = tf.nn.relu(pre_activation2, name=scope.name)

        _activation_summary(conv8)

    norm_8th = tf.nn.lrn(conv8, 1, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='8th_norm')

    # conv2d_transpose
    # concatenate along the axis of channels
    
    with tf.variable_scope('9th_Unsample_Transposed_conv2d') as scope:
        kernel1 = _variable_with_weight_decay('weights1', shape = [3, 3, 16, 32], stddev = 5e-2, wd = None)
        upcon9 = tf.nn.conv2d_transpose(norm_8th, kernel1, output_shape = tf.shape(conv1), strides = [1, 2, 2, 1], padding='SAME')
        upcon9 = tf.concat([upcon9, conv1], 3)

    upcon9 = tf.nn.dropout(upcon9, rate = 0.1)

    # 2@ConvLayers, 16@ 3x3 filters, Padding = 'same'

    with tf.variable_scope('9th_block_conv2d') as scope:
        kernel1 = _variable_with_weight_decay('weights1', shape = [3, 3, 32, 16], stddev = 5e-2, wd = None)
        conv = tf.nn.conv2d(upcon9, kernel1, [1, 1, 1, 1], padding='SAME')
        biases1 = _variable_on_cpu('biases1', [16], tf.constant_initializer(0.1))
        pre_activation1 = tf.nn.bias_add(conv, biases1)
        conv = tf.nn.relu(pre_activation1, name=scope.name)

        kernel2 = _variable_with_weight_decay('weights2', shape = [3, 3, 16, 16], stddev = 5e-2, wd = None)
        conv9 = tf.nn.conv2d(conv, kernel2, [1, 1, 1, 1], padding='SAME')
        biases2 = _variable_on_cpu('biases2', [16], tf.constant_initializer(0.1))
        pre_activation2 = tf.nn.bias_add(conv9, biases2)
        conv9 = tf.nn.relu(pre_activation2, name=scope.name)

        _activation_summary(conv9)

    norm_9th = tf.nn.lrn(conv9, 1, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='9th_norm')

    # 1@ConvLayers, 16@ 1x1 filters, Padding = 'same', activation = 'sigmoid'

    with tf.variable_scope('10th_block_conv2d') as scope:
        kernel1 = _variable_with_weight_decay('weights1', shape = [1, 1, 16, 1], stddev = 5e-2, wd = None)
        conv10 = tf.nn.conv2d(norm_9th, kernel1, [1, 1, 1, 1], padding='SAME')
        biases1 = _variable_on_cpu('biases1', [1], tf.constant_initializer(0.1))
        pre_activation1 = tf.nn.bias_add(conv10, biases1)
        conv10 = tf.nn.sigmoid(pre_activation1, name=scope.name)
        #conv10 = tf.math.multiply(conv10, 255, name=scope.name)

        _activation_summary(conv10)

    
    return conv10

def loss(images, labels):
    """a pixel-wise soft-max

    Add summary for "Loss" and "Loss/avg"
    Args:
        images: conv10 from inference()
        labels: Labels from distorted_inputs. A 4D tensor.

    Returns:
        Loss tensor of type float.

    """

    reshaped_images = tf.cast(tf.reshape(images, [-1]), tf.float32)
    reshape_labels = tf.cast(tf.reshape(labels, [-1]), tf.float32)
    reshape_labels = tf.math.divide(reshape_labels, 255)
    labels = tf.cast(reshape_labels, tf.float32)
    logits = tf.exp(reshaped_images) / tf.reduce_sum(tf.exp(reshaped_images))
    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets = labels, logits = logits, pos_weight = 1)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
    """Add summaries for losses in EF U-net model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
        total_loss: Total loss from loss().

    Returns:
        loss_averages_op: op for generating moving averages of losses.

    """
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def train(total_loss, global_step):
    """Train the U-net model

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
            processed.

    Returns:
        train_op: op for training.

    """
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
    tf.summary.scalar('learning_rate', lr)

    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    with tf.control_dependencies([apply_gradient_op]):
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    return variables_averages_op