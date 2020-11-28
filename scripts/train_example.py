import uuid
import json
import os
from topicc import train_topicc

params = {
    'model_type': 'TopicCEncSimpleBPemb',
    'model_params': {
        'embed_size': 300,
        'output_size': 30,
        'enc_hidden_size': 256
    },
    # 'model_params': {
    #     'embed_size': 300,
    #     'enc_hidden_size': 256,
    #     'attention_size': 128,
    #     'dense_size': 64,
    #     'output_size': 30
    # },
    'dataset_params': {
        'sequences_file': 'data/wikiVAlvl5_summaries.txt',
        'categories_file': 'data/wikiVAlvl5_categories.txt',
        'category_labels_file': 'data/wikiVAlvl5_category_labels.json'
    },
    'optimiser_params': {
        'epochs': 1,
        'batch_size': 32,
        'n_batch_validate': 100,
        'lr': 0.0001,
        'clip_grad': 10
    }
}


def main():
    train_topicc(params)


if __name__ == '__main__':
    main()
