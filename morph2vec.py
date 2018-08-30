# -*- encoding: utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import sys
import dynet_config
from dynet import *
import dynet
import numpy as np
from numpy.linalg import norm
from optparse import OptionParser
import dynet_config
import random
import utils
import numpy as np
import re

class learner:
    def __init__(self,options):
        self.model = ParameterCollection()
        random.seed(1)

        self.wdims = options.wembedding_dims
        self.mdims = options.membedding_dims

        self.morph_dict_array = utils.get_morph_dict_array(options.un_morph_path)
        self.morph_dict = utils.get_morph_dict(options.gold_morph_path)
        self.morph_gold = utils.get_morph_gold(self.morph_dict, self.morph_dict_array)
        self.m2i = self.create_m2i()

        self.sentences_original = self.read_data("special/clean-conllu.txt",True)
        self.w2i = self.vocab(self.sentences_original,10000)

        self.morph_lstm = [VanillaLSTMBuilder(1, self.mdims, self.wdims, self.model),
                                VanillaLSTMBuilder(1, self.mdims, self.wdims, self.model)]
        self.morph_hidLayer = self.model.add_parameters((self.wdims, self.wdims*2))
        self.morph_attW = self.model.add_parameters((self.wdims, self.wdims))
        self.morph_attV = self.model.add_parameters((1, self.wdims))
        self.word_attV = self.model.add_parameters((1, 1))
        self.mlookup = self.model.add_lookup_parameters((len(self.m2i), self.mdims))
        self.wlookup = self.model.add_lookup_parameters((len(self.w2i), self.wdims))

        self.trainer = AdamTrainer(self.model)
        self.trainer.set_sparse_updates(False)

        self.epochs = options.epochs


    def create_m2i(self):
        all_morphs = []
        m2i = {}
        m2i["UNK"] = 0
        for word in self.morph_dict_array.keys():
            for morphs in self.morph_dict_array[word]:
                all_morphs += morphs
        all_morphs = list(set(all_morphs))
        for idx in range(len(all_morphs)):
            m2i[all_morphs[idx]] = idx + 1
        return m2i

    def cosine_proximity(self, pred, gold):

        def l2_normalize(x):
            square_sum = dynet.sqrt(dynet.bmax(dynet.sum_elems(dynet.square(x)), np.finfo(float).eps * dynet.ones((1))[0]))
            return dynet.cdiv(x, square_sum)

        y_true = l2_normalize(pred)
        y_pred = l2_normalize(gold)
        return -dynet.sum_elems(dynet.cmult(y_true, y_pred))

    def pick_neg_log(self, pred, gold):
        return -dynet.log(dynet.pick(pred, gold))

    def binary_log_loss(self, pred, gold):
        y = dynet.scalarInput(gold)
        return dynet.binary_log_loss(pred, y)

    def __getSegmentationVector(self, morph_vec, seg_vec): #list of morph vectors and  segmentation attetion vectors
        seg_att = dynet.softmax(concatenate(seg_vec))
        seg_att_reshape = dynet.reshape(seg_att, (seg_att.dim()[0][0], 1))

        seg_morph = concatenate(morph_vec)
        seg_morph_reshape = dynet.reshape(seg_morph, (int(seg_morph.dim()[0][0]/self.wdims), self.wdims))

        morph_emb = dynet.sum_dim(dynet.cmult(seg_att_reshape,seg_morph_reshape), [0]) #weighted sum of morph vectors

        return morph_emb, seg_att

    def __getWordContext(self, word, context):
        return dynet.log_sigmoid(self.word_attV.expr() * dynet.dot_product(word,context))

    def __getMorphVector(self, word):
        return self.wlookup[self.w2i[word] if word in self.w2i else self.w2i["UNK"]]
        word_segs = self.morph_dict_array[word] if word in self.morph_dict_array else [["UNK"]]

        morph_lstm_forward, morph_lstm_backward = [], []
        morph_enc = []
        morph_vec, seg_vec = [], []
        for morph_seg in word_segs:
            seg = [self.m2i[morph] for morph in morph_seg]
            mlstm_forward = self.morph_lstm[0].initial_state()
            mlstm_backward = self.morph_lstm[1].initial_state()

            morph_lstm_forward.append(mlstm_forward.transduce([self.mlookup[m] for m in seg]))
            morph_lstm_backward.append(mlstm_backward.transduce([self.mlookup[m] for m in reversed(seg)]))

            morph_enc.append(concatenate([morph_lstm_forward[-1][-1],morph_lstm_backward[-1][-1]]))
            morph_vec.append(self.morph_hidLayer.expr() * morph_enc[-1]) #morph based word embedding for each segmentation
            seg_vec.append(self.morph_attV.expr() * dynet.tanh(self.morph_attW.expr() * morph_vec[-1])) #attention vector of segmentation
        morph_emb, seg_att = self.__getSegmentationVector(morph_vec, seg_vec) #weighted sum of segmentation embeddings and segmentation prediction

        return morph_emb

    def read_data(self,file_name,lower=False):
        data = ""
        with open(file_name) as text:
            for line in text:
                data += line.strip() + "\n"
        data = data.strip()

        symbols = ['.',',',';',':','-','(',')','?','!']
        data_clean = ""
        for line in data.split("\n"):
            for word in line.split(" "):
                for character in word:
                    if lower:
                        character = character.lower()
                    if character in symbols:
                        character = " " + character + " "
                    data_clean += character
                data_clean += " "
            data_clean = data_clean.strip()
            data_clean += "\n"

        data_clean = re.sub("\s{2,}", " ", data_clean)
        data_clean = data_clean.strip()

        sentences = []
        for line in data_clean.split("\n"):
            sentences.append(line.split(" "))
        return sentences

    def vocab(self,sentences, size):
        freq_dict = {}
        for sentence in sentences:
            for word in sentence:
                if word in freq_dict:
                    freq_dict[word] += 1
                else:
                    freq_dict[word] = 1
        w2i = {}
        w2i = {word: idx+1 for idx,word in enumerate(sorted(freq_dict, key=freq_dict.__getitem__, reverse=True)[:size])}
        w2i["UNK"] = 0

        return w2i

    def getSampling(self,sentence):
        window_size = 3
        valid_couples = []
        negative_couples = []
        word_list = list(self.w2i.keys())
        words = []

        for idx in range(len(sentence)-window_size+1):
            left,right = sentence[idx-int(window_size/2):idx], sentence[idx:idx+int(window_size/2)]
            word = sentence[idx]
            words += [word]
            for l,r in zip(left,right):
                valid_couples += [(word,l),(word,r)]
                words += [l,r]
            negative = word_list[random.randint(0,len(word_list)-1)]
            negative_couples += [(word,negative)]
            words += [negative]

        return valid_couples, negative_couples, list(set(words))

    def train(self):
        for i in range(self.epochs):
            sentences = self.sentences_original
            random.shuffle(sentences)
            mx = 0
            errs = []
            for sentence in sentences:
                valid_couples, negative_couples, words = self.getSampling(sentence)
                if mx != 0 and mx % 1000 == 0:
                    print(i)
                mx += 1

                morph_emb = {}
                for word in words:
                    morph_emb[word] = self.__getMorphVector(word)

                for couple in valid_couples:
                    word_emb = self.__getWordContext(morph_emb[couple[0]],morph_emb[couple[1]])
                    errs.append(self.binary_log_loss(word_emb, 1))

                for couple in negative_couples:
                    word_emb = self.__getWordContext(morph_emb[couple[0]],morph_emb[couple[1]])
                    errs.append(self.binary_log_loss(word_emb, 0))

                if len(errs)>0:
                    loss = (esum(errs))
                    loss.backward()
                    self.trainer.update()
                    errs = []
                dynet.renew_cg()
            print("Epoch Ends")

    def generate_word_vec(self,seq):
        seg = [self.m2i[morph] for morph in seq]
        mlstm_forward = self.morph_lstm[0].initial_state()
        mlstm_backward = self.morph_lstm[1].initial_state()

        morph_lstm_forward = mlstm_forward.transduce([self.mlookup[m] for m in seg])
        morph_lstm_backward = mlstm_backward.transduce([self.mlookup[m] for m in reversed(seg)])

        morph_enc = concatenate([morph_lstm_forward[-1],morph_lstm_backward[-1]])
        morph_vec = self.morph_hidLayer.expr() * morph_enc
        return morph_vec

    def can_generate_word_vector(self,seg):
        for morph in seg:
            if morph not in self.m2i:
                return False
        return True

    def morph2word(self, morph_dict):
        w2i = {}
        for word in morph_dict.keys():
            print(word)
            if self.can_generate_word_vector(morph_dict[word]):
                w2i[word] = self.generate_word_vec(morph_dict[word]).vec_value()
        return w2i

    def morph(self):
        morph_dict = {}
        for morph in self.m2i.keys():
            morph_dict[morph] = self.mlookup[self.m2i[morph]].vec_value()
        return morph_dict

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--goldmorph", dest="gold_morph_path", help="Path to Morph seqmentation file", metavar="FILE", default="N/A")
    parser.add_option("--unmorph", dest="un_morph_path", help="Path to Morph seqmentation file", metavar="FILE", default="N/A")
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--membedding", type="int", dest="membedding_dims", default=50)
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)
    parser.add_option("--prevectors", dest="external_embedding", help="Pre-trained vector embeddings", metavar="FILE")

    (options, args) = parser.parse_args()
    predictor = learner(options)
    for i in range(6):
        predictor.train()
        utils.save_embeddings("word_emb5.p",predictor.morph2word(utils.get_morph_dict("special/turkish_new_data_gold_segmented.txt",True)))
        utils.save_embeddings("morph_emb5.p",predictor.morph())


### To do:
# 1. Add l2 regularization (or add noise to the lookup parameters)
