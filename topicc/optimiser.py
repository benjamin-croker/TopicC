from typing import Tuple
import os

import torch
from torch.utils.data import DataLoader
from topicc import _TopicCBase, SeqCategoryDataset
from topicc.dataset import train_test_split


def evaluate_model(topicc: _TopicCBase, dataloader: DataLoader) -> Tuple[float, float]:
    # measures loss and accuracy

    train_state = topicc.training
    topicc.eval()

    n_samples = 0
    total_correct = 0
    total_loss = 0

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for sequences, labels in dataloader:
            log_prob = topicc(sequences)

            preds = topicc.predict(log_prob)
            n_samples += len(preds)
            total_correct += int(torch.eq(preds.to('cpu'), labels.to('cpu'), ).sum())

            total_loss += topicc.loss(log_prob, labels)

    if train_state:
        topicc.train()

    loss = total_loss / n_samples
    accuracy = total_correct / n_samples

    return loss, accuracy


def train(topicc_model: _TopicCBase, dataset: SeqCategoryDataset,
          model_id: str, checkpoint_dir: str,
          epochs=4, batch_size=32, n_batch_validate=100,
          lr=0.0001, clip_grad=10
          ) -> _TopicCBase:
    train_dataset, valid_dataset = train_test_split(dataset, test_prop=0.1)

    # note that dataloader will return a tuple of strings for the sequences
    # and a 1-D float tensor for the labels in each batch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4 * batch_size, shuffle=True)

    # set up the model for training
    topicc_model.use_device('cuda:0')
    topicc_model.train()

    optimizer = torch.optim.Adam(topicc_model.parameters(), lr=lr)

    best_valid_acc = 0

    print("train start")
    for i_epoch in range(epochs):
        # loss for each reporting block
        current_loss = 0
        n_samples = 0
        for i_batch, (sequences, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            log_prob = topicc_model(sequences)
            loss = topicc_model.loss(log_prob, labels)
            current_loss += loss
            n_samples += len(sequences)

            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(topicc_model.parameters(), clip_grad)
            optimizer.step()

            # index from 1 for counting number of batches
            if (i_batch + 1) % n_batch_validate == 0:
                print(f"epoch:{i_epoch + 1} batch:{i_batch + 1}")
                print(f"loss for last {n_batch_validate} batches: {current_loss / n_samples}")
                current_loss = 0
                n_samples = 0

                valid_loss, valid_acc = evaluate_model(topicc_model, valid_loader)
                print(f"validation loss: {valid_loss}")
                print(f"validation accuracy: {round(100 * valid_acc, 2)}% (best: {round(100 * best_valid_acc, 2)}%)")

                # save a checkpoint
                checkpoint = {
                    'model_state_dict': topicc_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, f'{model_id}-checkpoint.pt'))

                if valid_acc > best_valid_acc:
                    torch.save(checkpoint, os.path.join(checkpoint_dir, f'{model_id}-best.pt'))
                    best_valid_acc = valid_acc

    # load the best model
    checkpoint = torch.load(os.path.join(checkpoint_dir, f'{model_id}-best.pt'))
    topicc_model.load_state_dict(checkpoint['model_state_dict'])
    # exit training mode
    topicc_model.eval()

    return topicc_model
