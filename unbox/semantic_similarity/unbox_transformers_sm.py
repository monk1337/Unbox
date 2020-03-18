import scipy
from collections import Counter
from nltk.corpus import stopwords
from nltk import word_tokenize
stop = set(stopwords.words('english'))
from scipy import spatial
import numpy as np
from time import gmtime, strftime
import pandas as pd
import os
import logging
import operator
import pickle as pk
import json

# https://github.com/UKPLab/sentence-transformers
from sentence_transformers import SentenceTransformer

# download the model


class Unbox_transformers(object):
    
    
    # loading universal sentence encoder 
    # to find similar sentences/paragraphs
    
    def __init__(self, config = False ):
        
        default_confi = {'model_large': 'bert-large-nli-stsb-mean-tokens', 'model_m': 'bert-base-nli-mean-tokens'}
        
        if config:
            default_confi.update(default_confi)
        
        print("it's going to take some time, depend on your internet speed ... try base model if taking long time")
        self.model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
        print("model loaded")
        
        
    @staticmethod
    # cosine distance
    def similarity_score(vector_one, vector_2):
        result = 1 - spatial.distance.cosine(vector_one, vector_2)
        return result
    
    
    #preprocessing
    @staticmethod
    def preprocessing_stopwords(sentence):
        cleaned = [i.lower() for i in word_tokenize(sentence) if i not in stop]
        return " ".join(cleaned)
    
        #preprocessing
    @staticmethod
    def preprocessing_(sentence):
        cleaned = [i.lower() for i in word_tokenize(sentence)]
        return " ".join(cleaned)
    
    def unbox_transformer_sm(self, query, config_file = False ):
        
        default_config = {
                          'preprocessing'  :  False
                         }
        
        
        if config_file:
            default_config.update(config_file)
            
        
        if default_config['preprocessing']:
            
            query_vector_a =  Unbox_transformers.preprocessing_stopwords(query['query_a'].lower())
            query_vector_b =  Unbox_transformers.preprocessing_stopwords(query['query_b'].lower())
        else:
            query_vector_a =  query['query_a'].lower()
            query_vector_b =  query['query_b'].lower()
            

        # get the embeddings
        combine_query = [query_vector_a, query_vector_b ]
        emb_s         = self.model.encode(combine_query)
        
        # calculate the distance
        sm_value = Unbox_transformers.similarity_score(emb_s[0], emb_s[1])
        
        query['similarity_value'] = sm_value
        return query
    
    
    def get_transformer_vectors(self, query, config_file = False ):
        
        default_config = {
                          'preprocessing'  :  False,
                          'pickle'         :  True
                         }
        
        
        if config_file:
            default_config.update(config_file)
            
        
        if default_config['preprocessing']:
            query = [Unbox_transformers.preprocessing_stopwords(sentence.lower()) for sentence in query]
            
        
        # get the embeddings
        emb_s = self.model.encode(query)
        if default_config['pickle']:
            with open('./embeddings/embeddings_generated' +  str(strftime("%Y-%m-%d %H:%M:%S", gmtime())) ,'wb') as f:
                pk.dump(emb_s,f)
        
        
        
        output = {
                  'embeddings'      : np.array(emb_s), 
                  'total_queries'   : len(query), 
                  'total_embeddings': len(emb_s), 
                  'shape'           : np.array(emb_s).shape,
                  'queries'         : query
                 }
        
        return output