import torch
import torch.utils.data
from topicc import TopicC, WikiVALvl5Dataset
from topicc.data import train_test_split


def train(topicc: TopicC, dataset: WikiVALvl5Dataset) -> TopicC:
    # constants TODO: move to args
    lr = 0.0001
    epochs = 1000
    batch_size = 32
    clip_grad = 5
    report_batch = 10

    # set up the batch data
    # TODO: train/test split
    train_dataset, test_dataset = train_test_split(dataset, test_prop=0.001)
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    print(len(test_dataset))

    # set the model to training mode
    topicc.train()

    loss_fn = torch.nn.NLLLoss()

    optimizer = torch.optim.Adam(topicc.parameters(), lr=lr)
    current_loss = 0

    for i_epoch in range(epochs):
        for i_batch, (sequences, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            loss = loss_fn(topicc(sequences), labels)
            current_loss += loss
            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(topicc.parameters(), clip_grad)
            optimizer.step()

            if i_epoch % report_batch == 0:
                print(f"epoch:{i_epoch}  batch:{i_batch}  loss:{current_loss / report_batch}")
                # topicc.eval()
                # preds = topicc.predict(sequences[0:2])
                # topicc.train()
                current_loss = 0
                # examples = zip(
                #     sequences,
                #     dataset.labels_to_categories(labels[0:2]),
                #     dataset.labels_to_categories(preds)
                # )
                # for seq, actual, pred in examples:
                #     print(f"----\n{seq[0:100]} \nactual:{actual} | pred:{pred}\n----")

            if i_epoch % (report_batch*10) == 0:
                topicc.eval()
                preds = topicc.predict(sequences[0:5])
                topicc.train()
                examples = zip(
                    sequences,
                    dataset.labels_to_categories(labels[0:5]),
                    dataset.labels_to_categories(preds)
                )
                for seq, actual, pred in examples:
                    print(f"----\n{seq[0:100]} \nactual:{actual} | pred:{pred}\n----")

    # exit training mode
    topicc.eval()
    return topicc
