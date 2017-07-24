#encoding=utf-8
from text_mining import using_gensim
import os,pickle
from pprint import pprint
from gensim.models import Word2Vec
import numpy as np

def make_result_dir():
    try:
        os.mkdir('./result/')
        os.mkdir('./result/lda/')
        os.mkdir('./result/tfidf/')
        os.mkdir('./result/test/')
        os.mkdir('./result/test/bag/')
        os.mkdir('./result/test/w2v/')
    except:
        pass

def compute_topic_vector():
    w2v_model = Word2Vec.load('./result/w2v.model')
    data = pickle.load(open('./result/lda_topics.pkl','rb'))
    topic2vec={}
    for topic in data:
        print(20*'-')
        doc=sorted(data[topic].items(),key=lambda item:item[1],reverse=True)
        topic_vec=np.zeros(shape=(100))
        for word,weight in doc:
            # print word.decode('utf-8')+': '+str(weight)
            try:
                topic_vec+=weight*w2v_model.wv[word]
            except:
                print(word)
        topic2vec.setdefault(topic,topic_vec)
    pickle.dump(topic2vec, open('./result/topic_vectors.pkl', 'wb'))

make_result_dir()
# build the processing class
text_mine=using_gensim(min_bagsize=10)

#preprocessing such as word segmentation, removing stopwords and word count statistic
text_mine.preprocessing()

# build the vocabulary for your input corpus
text_mine.build_vocabulary()

# train tf-idf model
text_mine.train_tf_idf()

# train lda model
text_mine.train_lda(num_topics=100)

# train word2vec model
text_mine.train_Word2vec(size=100,window=5)

# generate topic vector
compute_topic_vector()
