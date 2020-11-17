import torch
import torch.utils.data
from topicc import TopicC, WikiVALvl5Dataset


def train(topicc: TopicC, dataset: WikiVALvl5Dataset) -> TopicC:
    # constants TODO: move to args
    lr = 0.005
    epochs = 1
    batch_size = 32
    clip_grad = 5
    report_batch = 100

    # set up the batch data
    # TODO: train/test split
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # set the model to training mode
    topicc.train()

    optimizer = torch.optim.Adam(topicc.parameters(), lr=lr)
    current_loss = 0
    for i_epoch in range(epochs):
        for i_batch, (sequences, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            loss = topicc.loss(sequences, labels)
            current_loss += loss
            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(topicc.parameters(), clip_grad)
            optimizer.step()

            if i_batch % report_batch == 0:
                print(f"epoch:{i_epoch}  batch:{i_batch}  loss:{current_loss / report_batch}")
                topicc.eval()
                preds = topicc.predict(sequences[0:2])
                topicc.train()
                current_loss = 0
                examples = zip(
                    sequences,
                    dataset.labels_to_categories(labels[0:2]),
                    dataset.labels_to_categories(preds)
                )
                for seq, actual, pred in examples:
                    print(f"----\n{seq[0:100]} \nactual:{actual} | pred:{pred}\n----")

    # exit training mode
    topicc.eval()
    return topicc
