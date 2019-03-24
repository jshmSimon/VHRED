import tensorflow as tf
import os


class BaseModel(object):
    def __init__(self, _scope):
        self._scope = _scope
        self.ops = {}

    def save(self, sess, ckpt_dir):
        print("Saving model...")
        self.saver.save(sess, ckpt_dir, self.global_step)
        print("Model saved")

    def load(self, sess, ckpt_dir):
        latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_dir))
        self.saver.restore(sess, latest_ckpt)
        print("Model loaded")
