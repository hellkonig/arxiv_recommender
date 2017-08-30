from urllib.request import urlopen
from bs4 import BeautifulSoup
import gensim
import pickle

# base api query url
base_url = "http://export.arxiv.org/api/query?"
# search parameters
search_query = 'id_list=1312.4543'

# open a connection to a URL using urllib2
weburl = urlopen(base_url+search_query)

# get the result code and print it
print("result code: " + str(weburl.getcode()))

# read the data from the URL and
data = weburl.read()

# parser the html
soup = BeautifulSoup(data,"html.parser")
#print(soup.prettify().encode('ascii','ignore'))
   
# retreve abstract
titles = soup.find_all('title')[0].get_text()
abstracts = soup.find_all('summary')[0].get_text()
print(abstracts)
paper_id = soup.find_all('id')[1].get_text().split('/')[4]

text_corpus = gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(abstracts), [paper_id])

print(text_corpus[0])

# loading the model
d2v_model = gensim.models.doc2vec.Doc2Vec.load('./result/d2v.model')

# 
inferred_vector = d2v_model.infer_vector(text_corpus[0])
print(d2v_model.docvecs.most_similar([inferred_vector]))
