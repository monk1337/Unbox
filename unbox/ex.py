from semantic_similarity import Unbox_tensorflow_hub
transformer_model = Unbox_tensorflow_hub()



embed = transformer_model.unbox_sm(query = { 'query_a': 'Hello World, how are you', 
                                             'query_b': 'I am fine' 
                                            }, 
                                  config_file = {'preprocessing': True})
print(embed)