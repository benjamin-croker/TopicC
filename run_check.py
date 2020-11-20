from topicc import data, model, optimiser


def main():
    print('start')
    # topicc = model.TopicCEncSimpleBPemb(embed_size=300, enc_hidden_size=256, output_size=30)
    topicc = model.TopicCEncBPemb(embed_size=300, enc_hidden_size=256, attention_size=128, dense_size=64, output_size=30)
    dataset = data.WikiVALvl5Dataset('data/summaries.txt', 'data/categories.txt', 'data/category_labels.json')
    topicc = optimiser.train(topicc, dataset)
    print('done')


if __name__ == '__main__':
    main()
