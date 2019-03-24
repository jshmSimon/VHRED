from base import BaseModel
from configs import args
from model.model_utils import create_multi_rnn_cell
import numpy as np

import tensorflow as tf

# encoder_RNN, context_RNN
class Encoder(BaseModel):
	def __init__(self, vocab_size, is_embedding=True):
		# if is_embedding=True, this is encoder_RNN
		# if is_embedding=False, this is context_RNN
		super().__init__('encoder')
		self.is_embedding = is_embedding
		with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
			self.embedding = tf.get_variable('lookup_table', [vocab_size, args.embed_dims])
			self.encoder_cell = create_multi_rnn_cell(args.rnn_type, args.rnn_size, args.keep_prob, args.num_layer)

	def __call__(self, inputs):
		with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
			if self.is_embedding:
				encoder_lengths = tf.count_nonzero(inputs, -1, dtype=tf.int32)
				embedded_inputs = tf.nn.embedding_lookup(self.embedding, inputs)
			else:
				encoder_lengths = None
				embedded_inputs = inputs
			outputs, states = tf.nn.dynamic_rnn(self.encoder_cell, embedded_inputs, encoder_lengths, dtype=tf.float32)
			return outputs, states

if __name__ == '__main__':
	sess = tf.Session()
	enc_inputs = np.array([[2,3,4],[3,4,5]])
	vocab_size = 18506
	encoder = Encoder(vocab_size)
	enc_inp_ph = tf.placeholder(tf.int32, [None, None])
	outputs, states = encoder(enc_inp_ph)

	# with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	result = sess.run(states, feed_dict={enc_inp_ph:enc_inputs})
	print(type(result))
	print(len(result))
	print(result[-1].shape)
	print(type(result[-1]))