from topicc import data, model
import torch
import torch.utils.data


def main():
    print('start')
    # A simple test to make sure everything's loading correctly
    topicc = model.TopicC(output_size=30, enc_hidden_size=200, attention_size=50, dense_size=100)
    dataset = data.WikiVALvl5Dataset('data/summaries.txt', 'data/categories.txt', 'data/category_labels.json')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

    for x, y in dataloader:
        print(model.topicc_loss(topicc(x), y))
        print(x[1])
        print(y)
        break

    print('done')


if __name__ == '__main__':
    main()
