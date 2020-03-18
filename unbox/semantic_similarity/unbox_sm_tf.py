# sentence similarity using tensorflow hub models

import tensorflow.compat.v1 as tf
#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.disable_eager_execution()
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
import tensorflow_hub as hub
logging.getLogger("tensorflow").setLevel(logging.WARNING)



class Unbox_tensorflow_hub(object):
    
    
    # loading universal sentence encoder 
    # to find similar sentences/paragraphs
    
    
#     model_dict = {
#                   "elmo": "https://tfhub.dev/google/elmo/" , 
#                   "use_m": "https://tfhub.dev/google/universal-sentence-encoder/",
#                   "use_l": "https://tfhub.dev/google/universal-sentence-encoder-large/"
#                   }
    
    def __init__(self):
        
        pass
        
    
    @staticmethod
    def load_model(model_type, version):
        
    #  model_dict[model_type] + str(version)
    # some tfhub models support '.load' and some '.Module' 

        if model_type == 'elmo':
            
            # elmo model 
            # version : 1, 2, 3
            module_url = "https://tfhub.dev/google/elmo/" + str(version)
            return hub.Module(module_url)
            
        elif model_type == 'use_m':
            sd_ver = ['1','2', 1, 2]
            if version in sd_ver:
                module_url = "https://tfhub.dev/google/universal-sentence-encoder/" + str(version)
                return hub.Module(module_url)
            else:

                module_url = "https://tfhub.dev/google/universal-sentence-encoder/" + str(version)
                return hub.load(module_url)
            
            # universal sentence encoder medium
            # version : 1 , 2 , 3 , 4
            
        elif model_type == 'use_l':
            
            sd_ver = ['1','2', '3', 1, 2,3 ]
            
            if version in sd_ver:
                module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/" + str(version)
                return hub.Module(module_url)
            else:

                module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/" + str(version)
                return hub.load(module_url)
            
            
            #universal sentence encoder large
            # version : 1 , 2 , 3 , 4, 5
    
    
    
    # input example 
    # dataset = ['Hello how are you', 'Hello I am fine']

    @staticmethod
    def get_embedding(model_type, version, dataset):
        
        model = Unbox_tensorflow_hub.load_model(model_type, version)
        
        similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
        similarity_message_encodings = model(similarity_input_placeholder)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())

            message_embeddings = session.run(similarity_message_encodings,
                                            feed_dict={similarity_input_placeholder: dataset})
            
        
        return message_embeddings
    
    
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
    
    
    
    
    @staticmethod
    def unbox_sm(query, config_file = False ):
        
        default_config = {
                          'model_type'     : 'use_l', 
                          'version'        : '5',
                          'preprocessing'  :  False
                         }
        
        
        if config_file:
            default_config.update(config_file)
            
        
        if default_config['preprocessing']:
            
            query_vector_a =  Unbox_tensorflow_hub.preprocessing_stopwords(query['query_a'].lower())
            query_vector_b =  Unbox_tensorflow_hub.preprocessing_stopwords(query['query_b'].lower())
        else:
            query_vector_a =  query['query_a'].lower()
            query_vector_b =  query['query_b'].lower()
            
        
        # get the embeddings
        combine_query = [query_vector_a, query_vector_b ]
        emb_s = Unbox_tensorflow_hub.get_embedding(default_config['model_type'], default_config['version'], combine_query)
        
        # calculate the distance
        sm_value = Unbox_tensorflow_hub.similarity_score(emb_s[0], emb_s[1])
        
        query['similarity_value'] = sm_value
        return query
    
    @staticmethod
    def get_vectors(query, config_file = False ):
        
        default_config = {
                          'model_type' : 'use_l', 
                          'version'    : '5' ,
                          'preprocessing'  :  False,
                          'pickle'         : True
                         }
        print(query)
        if config_file:
            default_config.update(config_file)
        
        if default_config['preprocessing']:
            query = [Unbox_tensorflow_hub.preprocessing_stopwords(sentence.lower()) for sentence in query]
            
        
        # get the embeddings
        emb_s = Unbox_tensorflow_hub.get_embedding(default_config['model_type'], default_config['version'], query)
        if default_config['pickle']:
            with open('embeddings_generated' +  str(strftime("%Y-%m-%d %H:%M:%S", gmtime())) ,'wb') as f:
                pk.dump(emb_s,f)
        
        
        
        output = {
                  'embeddings'      : np.array(emb_s), 
                  'total_queries'   : len(query), 
                  'total_embeddings': len(emb_s), 
                  'shape'           : np.array(emb_s).shape,
                  'queries'         : query
                 }
        
        return output
    
#     # pandas dataframe as output
#     @staticmethod
#     def get_bulk_vectors(query, config_file = False ):
        
#         default_config = {
#                           'model_type' : 'use_l', 
#                           'version'    : '5' ,
#                           'preprocessing'  : False,
#                           'pickle'         : True
#                          }
        
#         if config_file:
#             default_config.update(config_file)
        
        
#         if default_config['preprocessing']:
#             query = [preprocessing_stopwords(sentence.lower()) for sentence in query]
        
#         # get the embeddings
#         emb_s = get_embedding(default_config['model_type'], default_config['version'], query)
        
#         # save the output as pickle file
#         if default_config['pickle']:
#             with open('embeddings_generated' +  str(strftime("%Y-%m-%d %H:%M:%S", gmtime())) ,'wb') as f:
#                 pk.dump(emb_s,f)
                
        
#         output_ = {
#                   'embeddings'      : emb_s, 
#                   'total_queries'   : len(query), 
#                   'total_embeddings': len(emb_s), 
#                   'shape'           : np.array(emb_s).shape
#                   'queries'         : query
#                  }
        
#         return output_
    
    @staticmethod
    def chunks(lst, n):
        
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    
    @staticmethod
    def corpus_splitter(corpus, split_value, stop_words = False ):
        
        # split the corpus in paragraphs
        if stop_words:
            corpus = Unbox_tensorflow_hub.preprocessing_stopwords(sentence)
        else:
            corpus = Unbox_tensorflow_hub.preprocessing_(sentence)

        blob = TextBlob(corpus)
        sentences = [item.raw for item in blob.sentences]
        total_len = int(len(sentences)/split_value)
        
        chunk_sentences = list(Unbox_tensorflow_hub.chunks(sentences, split_value))
        return [" ".join(sen) for sen in chunk_sentences]