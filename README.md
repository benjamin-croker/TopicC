# TopicC
TopicC is a library of models for categorising text sequences.

## Training

Training is performed by `topicc.train_topicc(params)`. This function takes
a dictionary with all parameters specifying the model and training details.

If the Wikipedia Vital Article dataset has been downloaded,
the example model can be trained with:

```
$ python -m topicc.train example_params.json
```

Where the `example_params.json` file contains:

```json
{
  "model_type": "TopicCEncBPemb",
  "model_id": "example",
  "model_params": {
    "embed_size": 300,
    "enc_hidden_size": 256,
    "attention_size": 128,
    "dense_size": 64,
    "output_size": 30
  },
  "dataset_params": {
    "sequences_file": "data/wikiVAlvl5_summaries.txt",
    "categories_file": "data/wikiVAlvl5_categories.txt",
    "category_labels_file": "data/wikiVAlvl5_category_labels.json"
  },
  "optimiser_params": {
    "epochs": 4,
    "batch_size": 32,
    "n_batch_validate": 100,
    "lr": 0.0001,
    "clip_grad": 10
  }
}
```

Trained models are stored in the 'output/' folder, with a filename
matching the `model_id` parameter, and a `.topicc` extension.

## Using a trained model
After building the model, it can be used by loading the output file:

```
model = topicc.load_model('example.topicc')
model.predict("Example text")
```

See `scripts\predict_example.py`.

Below is a demonstations classifying some text from the
[Wikipedia Article on Suwon](https://en.wikipedia.org/wiki/Suwon).

This uses the model trained by `scripts\train_example.py`.

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

## Topic Classification Datasets
TopicC requires 3 files for training the model. These are specified
by `dataset_params`.

* `sequences_file`: Text file with one text sequence per line
* `categories_file`: Text file with the corresponding categories
* `category_labels_file`: JSON file mapping category names to integer labels

The script `scripts/download_wikiVAlvl5.py` will construct an example dataset
using the summaries of the
[English Wikipedia 50000 Vital Ariticles.](https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/5)

Run using
```
$ python -m scripts.download_wikiVAlvl5
```

The script will rate limit queries in compliance with the
[Wikipedia REST API](https://en.wikipedia.org/api/rest_v1/)
use. Please respect the limits.

## Wikipedia Article and Title Dataset
The blog post
["The Unknown Perils of Mining Wikipedia"](https://blog.lateral.io/2015/06/the-unknown-perils-of-mining-wikipedia/)
has a link to a dataset of the text and tiles of all Wikipedia articles
with at least 20 pag views from October 2013.

https://storage.googleapis.com/lateral-datadumps/wikipedia_utf8_filtered_20pageviews.csv.gz

If the extracted CSV file is placed in `data/documents_utf8_filtered_20pageviews.csv`,
then the script `scripts/process_wiki20views.py` will construct a dataset with
files for the articles and titles.

Run using
```
$ python -m scripts.process_wiki20views
```

In many cases the articles start with the title. These are removed from the article text.

## News Dataset

The Kaggle ["All the News"](https://www.kaggle.com/snapcrack/all-the-news)
contains ~ 150 000 news articles with their titles.

If the individual CSV files are extracted to `data/` then the script
`scripts/process_allTheNews.py` will construct a dataset with
files for the articles and titles.

Run using
```
$ python -m scripts.process_allTheNews
```

## Training Parameters

The following `optimiser_params` can be set. To use the defaults,
set `optimiser_params : {}` to an empty dictionary.

* `epochs`: Number of iterations through the training set
* `batch_size`: Number of examples per batch
* `n_batch_validate`: Evaluate performance on the validation set every n batches
* `lr`: Learning rate
* `clip_grad`: Gradient clipping to prevent large weight changes per batch