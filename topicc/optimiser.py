import torch
import torch.utils.data
from topicc import TopicC, WikiVALvl5Dataset


def train(topicc: TopicC, dataset: WikiVALvl5Dataset) -> TopicC:
    # constants TODO: move to args
    lr = 0.005
    epochs = 1
    batch_size = 32
    report_batch = 100

    # set up the batch data
    # TODO: train/test split
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # set the model to training mode
    topicc.train()

    optimizer = torch.optim.Adam(topicc.parameters(), lr=lr)

    for i_epoch in range(epochs):
        for i_batch, (sequences, labels) in enumerate(dataloader):

            # loss function should normalise by batch size?
            loss = topicc.loss(sequences, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"epoch:{i_epoch}  batch:{i_batch}  loss:{loss}")
            if i_batch % report_batch == 0:
                preds = topicc.predict(sequences[0:5])
                examples = zip(
                    dataset.labels_to_categories(labels[0:5]),
                    dataset.labels_to_categories(preds)
                )
                for actual, pred in examples:
                    print(f"actual:{actual} | pred:{pred}")

    # exit training mode
    topicc.eval()
    return topicc
