<p align="center">
  <img width="80" src="./Extra/unbox.png">
</p>
<h2 align="center">Unsupervised Toolbox</h2>



<p align="center">A micro framework for State of the Art Methods and models for unsupervised learning for NLU / NLG </p>


### Features

- Semantic similarity with context aware attention
- State of the art embeddings for text dataset
- From NLU to NLG : Question Generation
- Question Answering
- Deep Learning Clustering
- Autoencoders, Latent representation

#### still in development mode

## Natural Language Generation

1. Generate the questions from given Paragraph using unsupervised Transformers  in three lines


```python
from unbox import unbox_nlg

paragraph = ["Waiting had its world premiere at the \
              Dubai International Film Festival on 11 December 2015 to positive reviews \
              from critics. It was also screened at the closing gala of the London Asian \
              Film Festival, where Menon won the Best Director Award."]

result = unbox_nlg.generate_question(paragraph)

```
#### output

```python

{
    "Generated_result": [
        {
            "generated_questions": "Who won the Best Director Award ?",
            "answer": "Menon",
            "c.paragraph.text": "Waiting had its world premiere at the               Dubai International Film Festival on 11 December 2015 to positive reviews               from critics. It was also screened at the closing gala of the London Asian               Film Festival, where Menon won the Best Director Award."
        },
        {
            "generated_questions": "What did Menon do ?",
            "answer": "the Best Director Award",
            "c.paragraph.text": "Waiting had its world premiere at the               Dubai International Film Festival on 11 December 2015 to positive reviews               from critics. It was also screened at the closing gala of the London Asian               Film Festival, where Menon won the Best Director Award."
        }
    ]
}


```
## Question Answer

```python

from unbox.quention_answering import unbox_qa

print(unbox_qa({
    'question': "Who won the Best Director Award ?",
    'context': "Waiting had its world premiere at the \
              Dubai International Film Festival on 11 December 2015 to positive reviews \
              from critics. It was also screened at the closing gala of the London Asian \
              Film Festival, where Menon won the Best Director Award."
}))

```

#### output

```python

{'score': 0.9980590308245141, 'start': 250, 'end': 255, 'answer': 'Menon'}

```



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

#### output

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
   
#### output

```python

{
'query_a': 'Hello World, how are you', 
'query_b': 'I am fine', 
'similarity_value': 0.2853398323059082
}


```
