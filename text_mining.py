#encoding=utf-8

from gensim import corpora
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.ldamodel import LdaModel
from gensim.models import Word2Vec
from pprint import pprint
import pandas as pd
import os,re,jieba,pickle,time
from collections import defaultdict
import pickle

# jieba.load_userdict('./data/userdict.txt')
# print 'userdict loading finished'
from pprint import pprint
from six import iteritems

class using_gensim(object):
    def __init__(self,min_bagsize=10):
        self.train_path='./data/train/'
        self.bag_path='./result/test/bag/'
        self.wv2_path='./result/test/w2v/'
        self.min_bagsize=min_bagsize

    def fenci(self,dirname,n):
        stoplist = set([line.strip() for line in open('./data/stopword')])
        result=open('./result/test/bag/test'+str(n),'w')
        result1=open('./result/test/w2v/test'+str(n),'w')
        frequency = defaultdict(int)
        print(dirname)
        corpus_list = pickle.load(open(dirname,'rb'))
        for key, value in corpus_list.items():
            '''
            line=line.strip()
            line=line.split('\t')
            if len(line)!=2:
                print('error in '+dirname+' '+'line: '+str(i+1))
                os._exit(0)
            # print line[1]
            doc=re.sub(u'[^\u4e00-\u9fa5a-zA-Z]+','',line[1].decode('utf-8'))
            '''
            doc=value['abstract']
            seg = list(jieba.cut(doc))
            # for w in seg:
            #     if w in stoplist:
            #         print w
            if len(seg)>self.min_bagsize:
                w2v_text=b''
                for w in seg:
                    w2v_text+=(w+' ').encode('utf-8')
                print(w2v_text[:-1]+b'\n')
                result1.write((w2v_text[:-1]+b'\n').decode('utf-8')) # check double \n
            seg=[w for w in seg if len(w)>=2 and w not in stoplist]
            if len(seg)>self.min_bagsize:
                text=key.encode('utf-8')+b','
                for w in seg:
                    frequency[w] += 1
                    text+=(w+' ').encode('utf-8')
                result.write((text[:-1] + b'\n').decode('utf-8'))
        result.close()
        result1.close()
        result2=open('./result/frequency.csv','w')
        for w in frequency:
            result2.write((w+','+str(frequency[w])+'\n'))
        result2.close()

    def preprocessing(self,):
        for n,fname in enumerate(os.listdir(self.train_path)):
            print(fname)
            self.fenci(os.path.join(self.train_path, fname),n)

    def build_vocabulary(self,):
        stoplist = set([line.strip() for line in open('./data/stopword')])
        dictionary = corpora.Dictionary()
        for i, fname in enumerate(os.listdir(self.bag_path)):
            dictionary.add_documents(line.lower().split() for line in open(os.path.join(self.bag_path, fname)))
        stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
                    if stopword in dictionary.token2id]
        once_ids = []#[tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
        dictionary.filter_tokens(stop_ids + once_ids)
        dictionary.compactify()
        dictionary.save('./result/voc.dict')

    def train_tf_idf(self,):
        tic = time.clock()
        dictionary=corpora.Dictionary.load('./result/voc.dict')
        id2token = {dictionary.token2id[t]: t for t in dictionary.token2id}

        class MyCorpus(object):
            def __init__(self, path):
                self.path = path
            def __iter__(self,):
                for i, fname in enumerate(os.listdir(self.path)):
                    for line in open(os.path.join(self.path, fname)):
                        line=line.split(',')[1]
                        yield dictionary.doc2bow(line.lower().split())

        corpus_memory_friendly = MyCorpus(self.bag_path)
        print('training tfidf ...')
        tfidf = TfidfModel(corpus_memory_friendly)
        tfidf.save('./result/tf-idf.model')
        print('tfidf training finished')

        for i, fname in enumerate(os.listdir(self.bag_path)):
            tfidf_result = {}
            for line in open(os.path.join(self.bag_path, fname)):
                line = line.split(',')
                doc = dictionary.doc2bow(line[1].lower().split())
                if len(doc)>self.min_bagsize:
                    r_doc={id2token[i].encode('utf-8'):v for i,v in tfidf[doc]}
                    tfidf_result.setdefault(line[0],r_doc)
            pickle.dump(tfidf_result, open('./result/tfidf/test'+str(i)+'.pkl', 'wb'))
        print('train tf-idf time using = %2f min' % ((time.clock() - tic) / 60.))

    def train_lda(self,num_topics=100):
        tic = time.clock()
        dictionary=corpora.Dictionary.load('./result/voc.dict')
        id2token = {dictionary.token2id[t]: t for t in dictionary.token2id}

        class MyCorpus(object):
            def __init__(self, path):
                self.path = path
            def __iter__(self):
                for i, fname in enumerate(os.listdir(self.path)):
                    for line in open(os.path.join(self.path, fname)):
                        yield dictionary.doc2bow(line.lower().split())

        corpus_memory_friendly = MyCorpus(self.bag_path)
        print('training lda ...')
        lda = LdaModel(corpus=corpus_memory_friendly, id2word=dictionary, num_topics=num_topics, update_every=1, chunksize=10000, passes=1)
        lda.save('./result/lda.model')
        print('lda training finished')
        result={}
        for topic_id in range(num_topics):
            result.setdefault(topic_id,{id2token[word_id].encode('utf-8'): word_probability for word_id, word_probability in lda.get_topic_terms(topic_id)})
        pickle.dump(result, open('./result/lda_topics.pkl', 'wb'))

        for i, fname in enumerate(os.listdir(self.bag_path)):
            print(fname)
            result1 = {}
            for line in open(os.path.join(self.bag_path, fname)):
                line = line.split(',')[1]
                doc = dictionary.doc2bow(line.lower().split())
                result1.setdefault(line[0],{topic_id:topic_probability for topic_id, topic_probability in lda.get_document_topics(doc)})
            pickle.dump(result1, open('./result/lda/test' + str(i) + '.pkl', 'wb'))
        print('train lda time using = %2f min' % ((time.clock() - tic) / 60.))

    def train_Word2vec(self,size=100, window=5):
        tic = time.clock()
        class MySentences(object):
            def __init__(self, dirname):
                self.dirname = dirname

            def __iter__(self):
                for fname in os.listdir(self.dirname):
                    for line in open(os.path.join(self.dirname, fname)):
                        yield line.split()
        sentences = MySentences('./result/test/w2v/')
        print('Word2vec training ...')
        model = Word2Vec(sentences, size=size, window=window, min_count=5, workers=4)
        model.save('./result/w2v.model')
        print('train Word2vec time using = %2f min' % ((time.clock() - tic) / 60.))

if __name__ == "__main__":
    print('hello to gensim text mining')
