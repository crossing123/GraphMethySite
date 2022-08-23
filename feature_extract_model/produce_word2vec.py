# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:19:53 2020

@author: wanghongfei
"""

from importlib.resources import path
import numpy as np
from Bio import SeqIO
from nltk import trigrams, bigrams
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
import re
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

np.set_printoptions(threshold=np.inf)

aa_kind="lysine"
site_kind="K"
path="../seq_model/"+aa_kind+"/"
texts = []
for index, record in enumerate(SeqIO.parse(path+'me_withX_'+site_kind+'.fasta', 'fasta')):
    tri_tokens = bigrams(record.seq)
    temp_str = ""
    for item in ((tri_tokens)):
        #print(item),
        temp_str = temp_str + " " +item[0] + item[1]
        #temp_str = temp_str + " " +item[0]
    texts.append(temp_str)

seq=[]
stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
for doc in texts:
    doc = re.sub(stop, '', doc)
    seq.append(doc.split())

    
w2v_model = Word2Vec(seq, size=20, window=5, min_count=1, workers=8, sg=1)

vocab_list = list(w2v_model.wv.vocab.keys())

word_index = {word: index for index, word in enumerate(vocab_list)}

w2v_model.save(path+'withX_word2vec_'+site_kind+'.model')
print("word2vec_model is constructed")

w2v_model.wv.save_word2vec_format(path+'withX_word2vec_'+site_kind+'.vector')
