import gensim
import collections
import smart_open
import random
import os
from os import listdir
from os.path import isfile, join
import pickle
import multiprocessing

def read_corpus(fname,tokens_only=False):
    f = pickle.load(open(fname,'rb'))
    for index, key in enumerate(f):
        if tokens_only:
            yield gensim.utils.simple_preprocess(f[key]['abstract'])
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(f[key]['abstract']), [key])


train_corpus = list(read_corpus('./data/train/arxiv_daily.pkl'))

model = gensim.models.doc2vec.Doc2Vec(size=20,
	  window=8,
	  min_count=25,
	  dbow_words=1,
	  iter=5,
	  workers=max(1,multiprocessing.cpu_count()//2))

model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)

model.save('./result/d2v.model')

