from typing import Tuple, Union, Callable
import os

import torch
import torch.nn.utils.rnn as rnn
from torch.utils.data import DataLoader

from topicc import (
    _TopicCBase, _TopicKeyBase, CPU_DEVICE,
    SeqCategoryDataset, SeqKeywordsDataset, train_test_split
)


def evaluate_topicc_model(topicc: _TopicCBase, dataloader: DataLoader) -> Tuple[float, float]:
    # measures loss and accuracy on a validation set

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
            total_correct += int(torch.eq(preds.to(CPU_DEVICE), labels.to(CPU_DEVICE), ).sum())

            total_loss += topicc.loss(log_prob, labels)

    if train_state:
        topicc.train()

    loss = total_loss / n_samples
    accuracy = total_correct / n_samples

    return loss, accuracy


def evaluate_topickey_model(topicc_model: _TopicKeyBase, dataloader: DataLoader) -> Tuple[float, float]:
    train_state = topicc_model.training
    topicc_model.eval()

    n_samples = 0
    n_jaccard = 0
    total_loss = 0
    total_jaccard_score = 0

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for sequences, labels in dataloader:
            n_samples += len(labels)
            n_samples += 1

            model_output = topicc_model(sequences)
            total_loss += topicc_model.loss(model_output, labels)
            # get the padded labels calculated in the loss
            labels = topicc_model.cached_labels()
            
            preds = topicc_model.predict(model_output).bool()
            # this is not really correct as it's looking at the total batch
            # counts rather than set sizes per example, but it's much faster
            # and gives an idea of the training performance
            try:
                total_jaccard_score += (
                    torch.logical_and(labels, preds).sum().item() /
                    torch.logical_or(labels, preds).sum().item()
                )
            except ZeroDivisionError:
                pass

    if train_state:
        topicc_model.train()

    loss = total_loss / n_samples
    avg_jaccard_score = total_jaccard_score / n_jaccard

    return loss, avg_jaccard_score


def train(topicc_model: Union[_TopicCBase, _TopicKeyBase],
          dataset: Union[SeqCategoryDataset, SeqKeywordsDataset],
          model_id: str,
          checkpoint_dir: str,
          eval_fn: Callable[[Union[_TopicCBase, _TopicKeyBase], DataLoader], Tuple[float, float]],
          score_name: str,
          device=CPU_DEVICE,
          epochs=4,
          batch_size=32,
          n_batch_validate=100,
          lr=0.0001,
          clip_grad=10,
          validation_set_prop=0.1
          ) -> Union[_TopicCBase, _TopicKeyBase]:
    train_dataset, valid_dataset = train_test_split(dataset, test_prop=validation_set_prop)

    if hasattr(dataset, 'collate_fn'):
        collate_fn = dataset.collate_fn
    else:
        collate_fn = None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=4 * batch_size, shuffle=True, collate_fn=collate_fn
    )

    # set up the model for training
    topicc_model.use_device(device)
    topicc_model.train()

    optimizer = torch.optim.Adam(topicc_model.parameters(), lr=lr)

    best_v_score = 0

    print(f"train start on device:{device}")
    for i_epoch in range(epochs):
        # loss for each reporting block
        current_loss = 0
        n_samples = 0
        for i_batch, (sequences, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            model_output = topicc_model(sequences)
            loss = topicc_model.loss(model_output, labels)
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

                v_loss, v_score = eval_fn(topicc_model, valid_loader)
                print(f"validation loss: {v_loss}")
                print(f"validation {score_name}: {round(100 * v_score, 2)}% (best: {round(100 * best_v_score, 2)}%)")

                # save a checkpoint
                checkpoint = {
                    'model_state_dict': topicc_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, f'{model_id}-checkpoint.pt'))

                if v_score > best_v_score:
                    torch.save(checkpoint, os.path.join(checkpoint_dir, f'{model_id}-best.pt'))
                    best_v_score = v_score

    # load the best model
    checkpoint = torch.load(os.path.join(checkpoint_dir, f'{model_id}-best.pt'))
    topicc_model.load_state_dict(checkpoint['model_state_dict'])
    # exit training mode
    topicc_model.eval()

    return topicc_model
