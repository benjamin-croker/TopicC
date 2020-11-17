from topicc import data, model, optimiser
import torch
import torch.utils.data


def main():
    print('start')
    # A simple test to make sure everything's loading correctly
    topicc = model.TopicC(output_size=30, enc_hidden_size=128, attention_size=64, dense_size=128)
    dataset = data.WikiVALvl5Dataset('data/summaries.txt', 'data/categories.txt', 'data/category_labels.json')

    topicc = optimiser.train(topicc, dataset)

    print('done')


if __name__ == '__main__':
    main()
