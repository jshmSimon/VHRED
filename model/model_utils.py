import tensorflow as tf
from configs import args

def create_multi_rnn_cell(rnn_type, hidden_dim, keep_prob, num_layer):
	def single_rnn_cell():
		if rnn_type.lower() == "lstm":
			cell = tf.contrib.rnn.LSTMCell(hidden_dim)
		elif rnn_type.lower() == "gru":
			cell = tf.contrib.rnn.GRUCell(hidden_dim)
		else:
			raise ValueError(" # Unsupported rnn_type: %s." % rnn_type)
		cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
		return cell
	cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(num_layer)], state_is_tuple=True)
	# cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(num_layer)], state_is_tuple=False)
	return cell

def encoder(embedded_inputs, lengths, hparams, keep_prob):
	rnn_type = hparams.rnn_type
	hidden_dim = hparams.hidden_dim
	num_layer = hparams.num_layer

	encoder_cell = create_multi_rnn_cell(rnn_type, hidden_dim, keep_prob, num_layer)
	outputs, states = tf.nn.dynamic_rnn(
		cell=encoder_cell, inputs=embedded_inputs, sequence_length=lengths, dtype=tf.float32)

	return outputs, states
	# outputs: [batch_size, enc_max_len, hidden_dim]
	# states: ([batch_size, hidden_dim]) * num_layer

def draw_z_prior():
	return tf.truncated_normal([args.batch_size, args.latent_size])

def reparamter_trick(z_mean, z_logvar):
	z = z_mean + tf.exp(0.5 * z_logvar) * draw_z_prior()
	return z

def kl_weights_fn(global_step):
	return args.anneal_max * tf.sigmoid((10 / args.anneal_bias) * (
			tf.to_float(global_step) - tf.constant(args.anneal_bias / 2)))

def kl_loss_fn(mean_1, log_var_1, mean_2, log_var_2):
	return 0.5 * tf.reduce_sum(tf.exp(log_var_1) * tf.exp(log_var_2) +
	                           (mean_2 - mean_1) * tf.exp(log_var_2) * (mean_2 - mean_1) - 1 +
	                           log_var_2 - log_var_1) / tf.to_float(args.batch_size)