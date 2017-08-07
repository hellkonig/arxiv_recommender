import gensim
import collections
import smart_open
import random
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import os
from os import listdir
from os.path import isfile, join
import pickle

def nlp_clean(d):
    new_str = d.lower()
    dlist = tokenizer.tokenize(new_str)
    dlist = list(set(dlist).difference(stopword_set))
    return dlist

def read_corpus(fname, tokens_only=False):
    f = pickle.load(open(fname,'rb'))
    for index, key in enumerate(f):
        if tokens_only:
            yield nlp_clean(f[key]['abstract'])
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(nlp_clean(f[key]['abstract']), [index])

tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))

train_corpus = list(read_corpus('./data/train/arxiv_daily.pkl'))

model = gensim.models.doc2vec.Doc2Vec(size=100, window=5, min_count=5, workers=4)

model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)

ranks = []
second_ranks = []
for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)
    
    second_ranks.append(sims[1])

print(collections.Counter(ranks))

print('Document ({}): "{}"\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: "%s"\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))


