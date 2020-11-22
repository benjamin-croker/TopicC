from topicc import dataset, model, optimiser

params = {
    'model': {
        'embed_size':300,
        'enc_hidden_size':256,
        'attention_size':128,
        'dense_size':64,
        'output_size':30
    },
    'dataset': {
        'summaries_file': 'data/summaries.txt',
        'categories_file': 'data/categories.txt',
        'category_labels_file': 'data/category_labels.json'
    },
    'optimiser': {
        'epochs':4,
        'batch_size':32,
        'n_batch_validate':100,
        'lr':0.0001,
        'clip_grad':10
    }
}


def main():
    print('start')
    # topicc = model.TopicCEncSimpleBPemb(embed_size=300, enc_hidden_size=256, output_size=30)
    topicc = model.TopicCEncBPemb(**params['model'])
    wiki_va_lvl5 = dataset.WikiVALvl5Dataset(**params['dataset'])
    topicc = optimiser.train(topicc, wiki_va_lvl5, **params['optimiser'])
    print('done')


if __name__ == '__main__':
    main()
