import nltk
import os
import tensorflow as tf
from configs import args
import random
import numpy as np

# 讲语料中的一些词转为'<unk>'
def word_dropout(x):
    is_dropped = np.random.binomial(1, args.word_dropout_rate, x.shape)
    fn = np.vectorize(lambda x, k: args.UNK_ID if (
        k and (x not in range(4))) else x)
    return fn(x, is_dropped)

def read_data(filename):
    ret = []
    curPath = os.path.abspath(os.path.dirname(__file__))
    file = os.path.join(curPath, filename)
    with open(file, "r", encoding='utf-8') as f:
        for line in f:
            line = line.replace("\n", "")
            sents = line.split(" __eou__ ")
            conv = []
            for i in range(len(sents)):
                # 每个dialogue只取前4个utterance
                if i > 3:
                    break
                sent = sents[i]
                sent = sent.lower().strip()
                words = nltk.word_tokenize(sent)
                conv.append(words)
            # 剔除少于4个utterance的dialogue
            if len(conv) < 4:
                continue

            ret.append(conv)

    for dia in ret[:5]:
        print(len(dia))
        print(dia)
    return ret

def read_vocab(filename):
    word_list = []
    curPath = os.path.abspath(os.path.dirname(__file__))
    file = os.path.join(curPath, filename)
    with open(file, "r", encoding='utf-8') as f:
        for line in f:
            line = line.replace("\n", "").lower().strip()
            word_list.append(line)

    word_to_id = {}
    id_to_word = {}
    for i, w in enumerate(word_list):
        word_to_id[w] = i + 4
        id_to_word[i + 4] = w

    word_to_id["<pad>"] = 0
    word_to_id["<sos>"] = 1
    word_to_id["<eos>"] = 2
    word_to_id["<unk>"] = 3

    id_to_word[-1] = "-1"
    id_to_word[0] = "<pad>"
    id_to_word[1] = "<sos>"
    id_to_word[2] = "<eos>"
    id_to_word[3] = "<unk>"

    print(list(word_to_id.items())[:10])

    return word_to_id, id_to_word

def tokenize_data(data, word_to_id):
    unk_id = word_to_id["<unk>"]

    ret = []
    for conv in data:
        padded_conv = []
        for turn in conv:
            words = [word_to_id.get(w, unk_id) for w in turn]
            words = words[:min(args.max_len, len(words))]
            padded_conv.append(words)
        ret.append(padded_conv)

    print(ret[:5])
    return ret
    # List<List<List<int>>>

def ids_to_words(example, id_to_word, is_pre_utterance=True):
    result = []
    if is_pre_utterance:
        for utterance in example:
            sentence = ' '.join([id_to_word.get(id) for id in utterance if id > 0])
            result.append(sentence)
        return '-->'.join(result)
    else:
        return ' '.join([id_to_word.get(id) for id in example if id > 0])

def split_data(data):
    test_ratio = args.test_ratio
    num_all_examples = len(data)
    num_test = int(num_all_examples * test_ratio)
    random.shuffle(data)
    test_data = data[:num_test]
    train_data = data[num_test:]
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for dialogue in train_data:
        X_train.append(dialogue[:3]) # [[first utterance], [second utterance], [third utterance]]
        y_train.append(dialogue[3]) # [fourth utterance]
    for dialogue in test_data:
        X_test.append(dialogue[:3]) # [[first utterance], [second utterance], [third utterance]]
        y_test.append(dialogue[3]) # [fourth utterance]

    return X_train, y_train, X_test, y_test

def form_input_data(X, y):
    enc_inp = []
    dec_inp = []
    dec_out = []
    for dialogue in X:
        utterance_list = []
        for utterance in dialogue:
            utterance_list.append(np.array(utterance + [args.EOS_ID] + [args.PAD_ID] * (args.max_len - len(utterance))))
        enc_inp.append(np.array(utterance_list))
    for dialogue in y:
        dec_inp.append(np.array([args.SOS_ID] + dialogue + [args.PAD_ID] * (args.max_len - len(dialogue))))
        dec_out.append(np.array(dialogue + [args.EOS_ID] + [args.PAD_ID] * (args.max_len - len(dialogue))))
    enc_inp = np.array(enc_inp)
    dec_inp = np.array(dec_inp)
    dec_out = np.array(dec_out)

    return enc_inp, dec_inp, dec_out


