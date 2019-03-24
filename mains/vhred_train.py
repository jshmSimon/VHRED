from configs import args
from model import Encoder, VHRED
from data import VHREDDataLoader, ids_to_words

from tqdm import tqdm
import tensorflow as tf
import datetime
import math
import numpy as np

def main():
	sess = tf.Session()
	VHRED_dl = VHREDDataLoader(sess)
	VHRED_model = VHRED(dataLoader=VHRED_dl)

	sess.run(tf.global_variables_initializer())
	train_model(VHRED_model, VHRED_dl, sess, is_fresh_model=True)

	sess.close()

def sample_test(model, dataLoader, sess):
	for enc_inp, dec_inp, dec_tar in dataLoader.test_generator():
		infer_decoder_ids = model.infer_decoder_session(sess, enc_inp)
		sample_previous_utterance_id = enc_inp[:3]
		sample_infer_response_id = infer_decoder_ids[-1][:3]
		sample_true_response_id = dec_tar[:3]
		for i in range(len(sample_infer_response_id)):
			print('-----------------------------------')
			print('previous utterances:')
			print(ids_to_words(sample_previous_utterance_id[i], dataLoader.id_to_word, is_pre_utterance=True))
			print('true response:')
			print(ids_to_words(sample_true_response_id[i], dataLoader.id_to_word, is_pre_utterance=False))
			print('infer response:')
			print(ids_to_words(sample_infer_response_id[i], dataLoader.id_to_word, is_pre_utterance=False))
			print('-----------------------------------')
		break

def train_model(model, dataLoader, sess, is_fresh_model=True):
	if not is_fresh_model:
		model.load(sess, args.vhred_ckpt_dir)
	best_result_loss = 1000.0
	for epoch in range(args.n_epochs):
		print()
		print("---- epoch: {}/{} | lr: {} ----".format(epoch, args.n_epochs, sess.run(model.lr)))
		tic = datetime.datetime.now()

		train_batch_num = dataLoader.train_batch_num
		test_batch_num = dataLoader.test_batch_num

		loss = 0.0
		nll_loss = 0.0
		kl_loss = 0.0
		count = 0

		for (enc_inp, dec_inp, dec_tar) in tqdm(dataLoader.train_generator(), desc="training"):
			train_out = model.train_session(sess, enc_inp, dec_inp, dec_tar)

			count += 1
			global_step = train_out['global_step']
			loss += train_out['loss']
			nll_loss += train_out['nll_loss']
			kl_loss += train_out['kl_loss']

			if count % args.display_step == 0:
				current_loss = loss / count
				current_nll_loss = nll_loss / count
				current_kl_loss = kl_loss / count
				current_perplexity = math.exp(float(current_nll_loss)) if current_nll_loss < 300 else float("inf")
				print('Step {} | Batch {}/{} | Loss {} | NLL_loss {} | KL_loss {} | PPL {}'.format(global_step,
				                                                                                   count,
				                                                                                   train_batch_num,
				                                                                                   current_loss,
				                                                                                   current_nll_loss,
				                                                                                   current_kl_loss,
				                                                                                   current_perplexity))

		print()
		loss = loss / count
		nll_loss = nll_loss / count
		kl_loss = kl_loss / count
		perplexity = math.exp(float(nll_loss)) if nll_loss < 300 else float("inf")
		print('Train Epoch {}/{} | Loss {} | NLL_loss {} | KL_loss {} | PPL {}'.format(epoch,
		                                                                               args.n_epochs,
		                                                                               loss,
		                                                                               nll_loss,
		                                                                               kl_loss,
		                                                                               perplexity))


		test_loss = 0.0
		test_nll_loss = 0.0
		test_kl_loss = 0.0
		test_count = 0
		for (enc_inp, dec_inp, dec_tar) in tqdm(dataLoader.test_generator(), desc="testing"):
			test_out = model.test_session(sess, enc_inp, dec_inp, dec_tar)
			test_loss += test_out['loss_test']
			test_nll_loss += test_out['nll_loss_test']
			test_kl_loss += test_out['kl_loss']
			test_count += 1
		test_loss /= test_count
		test_nll_loss /= test_count
		test_kl_loss /= test_count
		test_perplexity = math.exp(float(test_nll_loss)) if test_nll_loss < 300 else float("inf")
		print('Test Epoch {}/{} | Loss {} | NLL_loss {} | KL_loss {} | PPL {}'.format(epoch,
		                                                                              args.n_epochs,
		                                                                              test_loss,
		                                                                              test_nll_loss,
		                                                                              test_kl_loss,
		                                                                              test_perplexity))

		print()

		print('# sample test')
		sample_test(model, dataLoader, sess)

		if test_loss < best_result_loss:
			model.save(sess, args.vhred_ckpt_dir)
			if np.abs(best_result_loss - test_loss) < 0.03:
				current_lr = sess.run(model.lr)
				sess.run(model.update_lr_op, feed_dict={model.new_lr: current_lr * 0.5})
			best_result_loss = test_loss
		toc = datetime.datetime.now()
		print(" # Epoch finished in {}".format(toc - tic))

if __name__ == '__main__':
    main()