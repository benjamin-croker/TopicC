# TopicC
TopicC is a library of models categorising text sequences.

## Training

Training is performed by `topicc.train_topicc(params)`. This function takes
a dictionary with all parameters specifying the model and training details.


For an example see `scripts/train_example.py`

```python
#scripts/train_example.py

from topicc import train_topicc

params = {
    'model_type': 'TopicCEncBPemb',
    'model_id': 'example',
    'model_params': {
        'embed_size': 300,
        'enc_hidden_size': 256,
        'attention_size': 128,
        'dense_size': 64,
        'output_size': 30
    },
    'dataset_params': {
        'sequences_file': 'data/wikiVAlvl5_summaries.txt',
        'categories_file': 'data/wikiVAlvl5_categories.txt',
        'category_labels_file': 'data/wikiVAlvl5_category_labels.json'
    },
    'optimiser_params': {
        'epochs': 4,
        'batch_size': 32,
        'n_batch_validate': 100,
        'lr': 0.0001,
        'clip_grad': 10
    }
}

def main():
    train_topicc(params, model_id='example')

if __name__ == '__main__':
    main()
```

The example model can be trained with 

```
$ python -m scripts.train_example
```

Note that the Wikipedia Vital Article dataset will need to be downloaded first.

## Using a trained model
After building the model, it can be used in interactive mode with the `predict_example` script.

Below is an example classifying some text from the
[Wikipedia Article on Suwon](https://en.wikipedia.org/wiki/Suwon).

```
$ python -m scripts.predict_example
init: TopicCEncBPemb model
Enter sequence
> Suwon is the capital and largest city of Gyeonggi-do, South Korea's most populous province which surrounds Seoul, the national capital. Suwon lies about 30 km (19 mi) south of Seoul. It is traditionally known as "The City of Filial Piety". With a population close to 1.3 million, it is larger than Ulsan, although it is not governed as a metropolitan city.Suwon has existed in various forms throughout Korea's history, growing from a small settlement to become a major industrial and cultural center. It is the only remaining completely walled city in South Korea. The city walls are one of the more popular tourist destinations in Gyeonggi Province. Samsung Electronics R&D center and headquarters are in Suwon.
Predictions: ['Geography_Cities', 'Geography_Countries']
```

## Model Types

The current models are: `TopicCEncBPemb`, `TopicCEncSimpleBPemb` and `TopicCDenseSpacy`.

### TopicCEncBPemb
The `TopicCEncBPemb` model uses the English model [BPEmb](https://nlp.h-its.org/bpemb/)
byte-pair-encoded word embeddings by Benjamin Heinzerling and Michael Strube.

These are fed into a bi-directional LSTM encoder with attention and a dense
layer before the output.

This model requires the following `model_params`.

* `embed_size`: word embedding vector size, must match the dim of an available BPEmb model
* `enc_hidden_size`: Dimenion of the LSTM encoder
* `attention_size`: Dimension of the attention vector
* `dense_size`: Dimension of the dense layer
* `output_size`: number of possible categories

### TopicCEncSimpleBPemb
The `TopicCEncSimpleBPemb` model uses the English model [BPEmb](https://nlp.h-its.org/bpemb/)
byte-pair-encoded word embeddings by Benjamin Heinzerling and Michael Strube.

These are fed into a bi-directional GRU encoder, then into the output.

This model requires the following `model_params`.

* `embed_size`: word embedding vector size, must match the dim of an available BPEmb model
* `enc_hidden_size`: Dimenion of the GRU encoder
* `output_size`: number of possible categories

### TopicCDenseSpacy
The `TopicCDenseSpacy` model uses the `en_core_web_lg` [spaCy](https://spacy.io/) 
model to construct a single vector for each text sequence.

This is fed through two fully connected dense layers, into the output.

This model requires the following `model_params`.

* `embed_size`: word embedding vector size, must match the dim of the `en_core_web_lg` model
* `enc_hidden_size`: Dimenion of the LSTM encoder
* `attention_size`: Dimension of the attention vector
* `dense_size`: Dimension of the dense layer
* `output_size`: number of possible categories