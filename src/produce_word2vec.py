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
import argparse

os.environ["CUDA_VISIBLE_DEVICES"]="0"
np.set_printoptions(threshold=np.inf)
aa_kind="lysine"
# site_kind="K"


def Produce(site_kind='K'):
    #Tranform the format to fasta
    name_seq=np.load(f"../dat/data_example/{site_kind}_site/me_name_seq_{site_kind}_have_PDB_alphafold.npy",allow_pickle=True).item()
    os.makedirs(f'../dat/data_example/{site_kind}_site/feature_model',exist_ok=True)
    path=f"../dat/data_example/{site_kind}_site/feature_model/"
    os.makedirs(path,exist_ok=True)
    with open(f'../dat/data_example/{site_kind}_site/feature_model/me_withX_{site_kind}.fasta','w') as f:
        for name,seq in name_seq.items():
            f.write('>'+name+'\n')
            f.write('X'+seq+'X'+'\n')
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

    # Train the Word2Vec model   
    w2v_model = Word2Vec(seq, size=20, window=5, min_count=1, workers=8, sg=1)
    vocab_list = list(w2v_model.wv.vocab.keys())
    word_index = {word: index for index, word in enumerate(vocab_list)}
    w2v_model.save(f'{path}withX_word2vec_{site_kind}.model')
    print("word2vec_model is constructed")
    w2v_model.wv.save_word2vec_format(f'{path}withX_word2vec_{site_kind}.vector')


def get_params():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-t','--sitetype', type=str, default='K',help="The kind of amino acid")
    args = parser.parse_args()
    return args


def Run(args):
    Produce(site_kind=args['sitetype'])

if __name__ == '__main__':
    args=vars(get_params())
    Run(args)
