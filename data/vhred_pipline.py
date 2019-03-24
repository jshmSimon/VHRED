from data.data_process import read_data, read_vocab, tokenize_data, split_data, form_input_data
from configs import args
import numpy as np

class VHREDDataLoader(object):
    def __init__(self, sess):
        self.sess = sess

        #load data
        self.raw_data = read_data(filename='samples.txt')
        self.word_to_id, self.id_to_word = read_vocab(filename='vocab.txt')
        self.vocab_size = len(self.word_to_id)
        # 单词转为数字
        self.data = tokenize_data(self.raw_data, self.word_to_id)

        # 划分数据集
        X_train, y_train, X_test, y_test = split_data(self.data)
        self.train_size = len(X_train)
        self.test_size = len(X_test)
        self.train_batch_num = self.train_size // args.batch_size
        self.test_batch_num = self.test_size // args.batch_size


        # 形成encoder_input, decoder_input, decoder_output
        self.enc_inp_train, self.dec_inp_train, self.dec_out_train = form_input_data(X_train, y_train)
        self.enc_inp_test, self.dec_inp_test, self.dec_out_test = form_input_data(X_test, y_test)
        # print(type(enc_inp_test[3][0]))
        # print(enc_inp_test[3])

        # self.train_iterator, self.train_init_dict = data_pipline(enc_inp_train, dec_inp_train, dec_out_train, self.sess)
        # self.test_iterator, self.test_init_dict = data_pipline(enc_inp_test, dec_inp_test, dec_out_test, self.sess)

    def train_generator(self):
        for i in range(self.train_batch_num):
            start_index = i * args.batch_size
            end_index = min((i + 1) * args.batch_size, self.train_size)
            encoder_inputs = self.enc_inp_train[start_index: end_index]
            decoder_inputs = self.dec_inp_train[start_index: end_index]
            decoder_targets = self.dec_out_train[start_index: end_index]
            yield encoder_inputs, decoder_inputs, decoder_targets

    def test_generator(self):
        for i in range(self.test_batch_num):
            start_index = i * args.batch_size
            end_index = min((i + 1) * args.batch_size, self.test_size)
            encoder_inputs = self.enc_inp_test[start_index: end_index]
            decoder_inputs = self.dec_inp_test[start_index: end_index]
            decoder_outputs = self.dec_out_test[start_index: end_index]
            yield encoder_inputs, decoder_inputs, decoder_outputs


if __name__ == '__main__':
    loader = VHREDDataLoader(sess='as')
    print('--------------------------------------')
    print(loader.dec_inp_train.shape)
    print(loader.dec_inp_train[0].shape)
    print(loader.dec_inp_train[:2])
    print(loader.dec_inp_train[0])

