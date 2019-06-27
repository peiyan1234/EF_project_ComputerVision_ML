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

tf.app.flags.DEFINE_string('show_dir', '/home/alvinli/Desktop/EF/dataset/EF-training-Pool/show',"""Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',"""Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/alvinli/Desktop/EF/dataset/EF-training-Pool/train',"""Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 1,"""How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 1,"""Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,"""Whether to run eval only once.""")

##### Notice !!
##### Please change batch_size to 1 from Unet.py 

def show():
    """
    show the predicted mask for the segmentation purpose. 
    """
    
    with tf.Graph().as_default() as g:
      eval_data = FLAGS.eval_data == 'test'
      images, labels = Unet.inputs(eval_data=eval_data)
      logits = Unet.inference(images)
      logits = tf.reshape(logits, [1, height * width])
      labels = tf.math.divide(labels, 255)
      labels = tf.reshape(labels, [1, height * width])
      Overlap_rate = tf.multiply(logits, labels)
      Overlap_rate = tf.math.divide(tf.reduce_sum(Overlap_rate, 1), tf.reduce_sum(labels, 1))
      variable_averages = tf.train.ExponentialMovingAverage(Unet.MOVING_AVERAGE_DECAY)
      variables_to_restore = variable_averages.variables_to_restore()
      saver = tf.train.Saver(variables_to_restore)

      summary_op = tf.summary.merge_all()

      summary_writer = tf.summary.FileWriter(FLAGS.show_dir, g)

      while True:
        eval_once(saver, summary_writer, Overlap_rate, summary_op, logits, labels, images)
        if FLAGS.run_once:
          break
        time.sleep(FLAGS.eval_interval_secs)


def eval_once(saver, summary_writer, Overlap_rate, summary_op, logits, labels, images):

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

        step = 0
        true_count = 0  
        while step < 1 and not coord.should_stop():
          predictions = sess.run(Overlap_rate)
          true_count += np.sum(predictions)

          #original_img = sess.run(images)
          #original_img = np.reshape(original_img, (600, 800))

          predicted_image = sess.run(logits)
          predicted_image = np.reshape(predicted_image, (width, height))
          ground_truth = sess.run(labels)
          ground_truth = np.reshape(ground_truth, (width, height))
          step += 1

        precision = true_count / 1
        print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
        #original_img = Image.fromarray(np.uint8(original_img * 255))
        #original_img.show()
        predicted_image = Image.fromarray(np.uint8(predicted_image * 255)) 
        predicted_image.show()
        ground_truth = Image.fromarray(np.uint8(ground_truth * 255)) 
        ground_truth.show()

        summary = tf.Summary()
        summary.ParseFromString(sess.run(summary_op))
        summary.value.add(tag='Precision @ 1', simple_value=precision)
        summary_writer.add_summary(summary, global_step)
      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

def main(argv=None):
    if tf.gfile.Exists(FLAGS.show_dir):
      tf.gfile.DeleteRecursively(FLAGS.show_dir)
    tf.gfile.MakeDirs(FLAGS.show_dir)
    show()

if __name__ == '__main__':
    tf.app.run()
