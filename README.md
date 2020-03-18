<p align="center">
  <img width="80" src="./Extra/unbox.png">
</p>
<h2 align="center">Unsupervised Toolbox</h2>



<p align="center">A micro framework for State of the Art Methods and models for unsupervised learning on Natural language processing</p>


### Features

- Semantic similarity with context aware attention
- State of the art embeddings for text dataset
- From NLU to NLG : Question Generation
- Question Answering
- Deep Learning Clustering
- Autoencoders, Latent representation


## Semantic similarity


### From Pre-trained Transformers
1. Get sentence similaity from Pre-trained Transformers in just three lines


```python
from unbox_transformers_sm import Unbox_transformers
transformer_model = Unbox_transformers()

embed = transformer_model.unbox_transformer_sm(query = {
                                                        'query_a': 'Hello how are you', 
                                                        'query_b': 'Hello I am fine' 
                                                        }, 
                                               config_file = {'preprocessing': True})
```


```python

{
'query_a': 'Hello how are you', 
'query_b': 'Hello I am fine', 
'similarity_value': 0.6853398323059082
}


```



### From Tensorflow-hub 
2. Get sentence similaity from Pre-trained Tensorflow-hub in just three lines


```python
from semantic_similarity import Unbox_tensorflow_hub
transformer_model = Unbox_tensorflow_hub()



embed = transformer_model.unbox_sm(query = { 'query_a': 'Hello World, how are you', 
                                             'query_b': 'I am fine' 
                                            }, 
                                  config_file = {'preprocessing': True})
```
   
   
```python

{
'query_a': 'Hello World, how are you', 
'query_b': 'I am fine', 
'similarity_value': 0.2853398323059082
}


```
