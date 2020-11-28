import uuid
import json
import os
from topicc import dataset, model, optimiser
from topicc.model import save_topicc

params = {
    'model': {
        'embed_size': 300,
        'enc_hidden_size': 256,
        'attention_size': 128,
        'dense_size': 64,
        'output_size': 30
    },
    'dataset': {
        'summaries_file': 'data/summaries.txt',
        'categories_file': 'data/categories.txt',
        'category_labels_file': 'data/category_labels.json'
    },
    'optimiser': {
        'epochs': 1,
        'batch_size': 32,
        'n_batch_validate': 100,
        'lr': 0.0001,
        'clip_grad': 10
    }
}


def main():
    run_id = str(uuid.uuid1())
    filename = os.path.join('output', f'{run_id}-params.json')

    with open(filename, 'w') as f:
        json.dump(params, f, indent=2)

    print(f'start: {run_id}')

    topicc = model.TopicCEncBPemb(**params['model'])
    wiki_va_lvl5 = dataset.WikiVALvl5Dataset(**params['dataset'])
    topicc = optimiser.train(
        topicc, wiki_va_lvl5, run_id, 'checkpoints', **params['optimiser']
    )
    save_topicc(topicc, os.path.join('output', f'{run_id}.topicc'))

    print('done')


if __name__ == '__main__':
    main()
