from topicc import data, model, optimiser
import torch
import torch.utils.data


def main():
    print('start')
    # A simple test to make sure everything's loading correctly
    # topicb = model.TopicB(embed_size=300, hidden1_size=500, hidden2_size=100, output_size=30)
    topicc = model.TopicC(embed_size=300, enc_hidden_size=256, output_size=30)
    dataset = data.WikiVALvl5Dataset('data/summaries.txt', 'data/categories.txt', 'data/category_labels.json')

    topicc = optimiser.train(topicc, dataset)

    print('done')


if __name__ == '__main__':
    main()
