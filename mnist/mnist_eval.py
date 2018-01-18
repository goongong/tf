import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import minist_train

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
  with tf.Graph().as_default as g:
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    
		validation_feed = {x:mnist.validation.images, y_:mnist.validation.labels}

		y = mnist.inference(x, None)

		correction_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(correction_prediction, tf.float32)

		variable_average = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
		variable_average_restore = variable_average.variable_to_restore()
		saver = tf.train.Saver(variable_average_restore)

		while 1:
			with tf.Session() as sess:
			  ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(sess.ckpt.model_checkpoint_path)
					global_step
					accuracy_score = sess.run(accuracy, feed_dict=validation_feed)
					print()
				else:
					print('')
					return
				time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None)
	mnist = 
	evaluate(mnist)

if __name__ == "__main__":
	tf.app.run()

