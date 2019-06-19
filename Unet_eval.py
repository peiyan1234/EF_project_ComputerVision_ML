from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

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

def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run(top_k_op)
        true_count += np.sum(predictions)
        step += 1

      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

def evaluate():
  """Eval small_project for a number of steps."""
  with tf.Graph().as_default() as g:
    eval_data = FLAGS.eval_data == 'test'
    images, labels = Unet.inputs(eval_data=eval_data)

    logits = Unet.inference(images)
    logits = tf.reshape(logits, [FLAGS.batch_size, height * width])
    
    labels = tf.math.divide(labels, 255)
    labels = tf.reshape(labels, [FLAGS.batch_size, height * width])

    #top_k_op = tf.nn.in_top_k(logits, tf.cast(labels, tf.int32), height * width)
    top_k_op = tf.multiply(logits, labels)
    denominator = tf.reduce_sum(tf.square(logits), 1) * tf.reduce_sum(tf.square(labels), 1)
    top_k_op = tf.math.divide(tf.reduce_sum(top_k_op, 1), tf.sqrt(denominator))

    variable_averages = tf.train.ExponentialMovingAverage(Unet.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()