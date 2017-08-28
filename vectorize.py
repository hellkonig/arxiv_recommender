import gensim
import pickle

# loading the model
d2v_model = gensim.models.doc2vec.Doc2Vec.load('./result/d2v.model')

# 
inferred_vector = d2v_model.infer_vector(['Reproducing', 'the', 'inefficiency', 'of', 'galaxy', 'formation', 'across', 'cosmic', 'time', 'with', 'a', 'large', 'sample', 'of', 'cosmological', 'hydrodynamical', 'simulations'])
print(d2v_model.docvecs.most_similar([inferred_vector]))
