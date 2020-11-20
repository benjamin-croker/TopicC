from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.utils.rnn as rnn

from bpemb import BPEmb
import spacy


class TopicC(nn.Module):
    def __init__(self):
        super(TopicC, self).__init__()
        self._device = 'cpu'

    def use_device(self, device):
        self.to(device)
        self._device = device

    def forward(self, sequences: List[str]):
        # Must return a log_softmax
        pass

    def embed_sequence(self, sequence: str):
        # returns a list of sequence vectors
        pass

    def loss(self, log_prob: torch.Tensor, labels: torch.Tensor):
        labels = labels.to(self._device)
        return nn.functional.nll_loss(log_prob, labels, reduction='sum')

    @staticmethod
    def predict(log_prob: torch.Tensor) -> torch.Tensor:
        _, pred = log_prob.topk(1)
        return pred.squeeze()


class TopicCDenseSpacy(TopicC):
    def __init__(self,
                 embed_size,
                 hidden1_size,
                 hidden2_size,
                 output_size):
        super(TopicCDenseSpacy, self).__init__()
        print("init TopicCDenseSpacy model...")

        # todo: factor out params
        self.embedding_model = spacy.load("en_core_web_lg", disable=['tagger', 'parser', 'ner'])
        print("loaded language model")
        self.in_to_h1 = nn.Linear(embed_size, hidden1_size)
        self.h1_to_h2 = nn.Linear(hidden1_size, hidden2_size)
        self.h2_to_out = nn.Linear(hidden2_size, output_size)

    def embed_sequences(self, sequences: List[str]) -> torch.Tensor:
        return torch.tensor(
            [self.embedding_model(sequence).vector for sequence in sequences]
        ).to(self._device)

    def forward(self, sequences: List[str]) -> torch.Tensor:
        # list of Tensors of dim seq_len, embed_size
        seq_vecs = self.embed_sequences(sequences)

        h1 = torch.tanh(self.in_to_h1(seq_vecs))
        h2 = torch.tanh(self.h1_to_h2(h1))
        output = self.h2_to_out(h2)

        return nn.functional.log_softmax(output, dim=1)


class TopicCEncBPemb(TopicC):
    def __init__(self,
                 embed_size,
                 output_size,
                 enc_hidden_size,
                 attention_size,
                 dense_size):
        super(TopicCEncBPemb, self).__init__()
        print("init TopicCEncBPemb model...")

        self.embedding_model = BPEmb(dim=embed_size, lang="en", vs=100000)
        self.encoder = nn.LSTM(
            input_size=embed_size,
            hidden_size=enc_hidden_size,
            num_layers=2,
            bidirectional=True
        )
        # 2* since LSTM is bi-directional
        self.enc_to_att_map = nn.Linear(2 * enc_hidden_size, attention_size, bias=False)
        self.att_to_pointer_map = nn.Linear(attention_size, 1, bias=False)
        self.seq_to_dense_map = nn.Linear(4 * enc_hidden_size, dense_size)
        self.dense_to_output_map = nn.Linear(dense_size, output_size)

    def embed_sequence(self, sequence: str) -> torch.Tensor:
        v_ids = self.embedding_model.encode_ids(sequence)
        return torch.tensor(self.embedding_model.vectors[v_ids]).to(self._device)

    def create_seq_vecs(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # returns the padded seq vector, lengths and original order index
        # sequences are sorted by length, and can be reverted to their original
        # order with the unsorting index vector

        # start by embedding the sequence vectors
        seq_vecs = [self.embed_sequence(s) for s in sequences]

        # sort lengths and set the device
        lengths = torch.tensor([seq_v.shape[0] for seq_v in seq_vecs])
        lengths, sort_i = lengths.sort(descending=True)
        _, orig_i = sort_i.sort()

        # pad the seq vecs and sort by length (dim 2 is the batch dimension)
        seq_vec_pad = rnn.pad_sequence(seq_vecs).to(self._device)
        seq_vec_pad = seq_vec_pad[:, sort_i, :]

        return seq_vec_pad, lengths, orig_i

    def forward(self, sequences: List[str]) -> torch.Tensor:
        # Make the word embeddings for each sequence
        pad_seq_vecs, lengths, orig_i = self.create_seq_vecs(sequences)

        # pack the sequence for the GRU
        # packed_seq_vecs.shape = max_seq_len, batch_size, embedding_dim
        packed_seq_vecs = rnn.pack_padded_sequence(pad_seq_vecs, lengths)

        # run through the GRU
        enc_outputs, (h_n, _) = self.encoder(packed_seq_vecs)
        # unpack the sequence
        # enc_outputs.shape = max_seq_len, batch_size, 2*enc_hidden_size
        enc_outputs, _ = nn.utils.rnn.pad_packed_sequence(enc_outputs)

        # # encoder masks to indicate which parts of the sequence should be considered
        enc_masks = torch.zeros(
            enc_outputs.shape[0], enc_outputs.shape[1], 1,
            dtype=torch.bool, device=self._device
        )
        for i, length in enumerate(lengths):
            enc_masks[length:, i, 0] = True

        # att_vec.shape = max_seq_len, batch_size, attention_size
        att_vec = self.enc_to_att_map(enc_outputs)
        att_vec = torch.tanh(att_vec)
        # pointer to the where attention is given to each index of the sequence
        # pointer.shape = max_seq_len, batch_size, 1
        pointer = self.att_to_pointer_map(att_vec)
        # mask out sections which are not part of the sequence
        pointer = pointer.masked_fill(enc_masks, -float('inf'))
        # pointer_w.shape = max_seq_len, batch_size, 1
        # each slice pointer_w[:, seq_n, 0] will sum to 1, where the values
        # indicate where the most attention should be paid
        pointer_w = nn.functional.softmax(pointer, dim=0)

        # permute so the batch dimension is first instead of second
        # pointer_w.shape = batch_size, max_seq_len, 1
        pointer_w = pointer_w.permute(1, 0, 2)
        # permute so the batch dimension is first, and seq_len dim is summed
        # enc_outputs.shape = batch_size, 2*enc_hidden_size, max_seq_len
        enc_outputs = enc_outputs.permute(1, 2, 0)
        # weighted sum of states
        # att_output.shape = batch_size, 2*enc_hidden_size, 1
        att_output = torch.bmm(enc_outputs, pointer_w)
        # remove the last dimension
        # att_output.shape = batch_size, 2*enc_hidden_size
        att_output = att_output.squeeze(dim=2)

        # combine with the hidden states
        seq_output = torch.cat((att_output, h_n[0], h_n[1]), dim=1)

        # have a dense non-linear layer
        # dense.shape = batch_size, dense_size
        dense = self.seq_to_dense_map(seq_output)
        dense = torch.tanh(dense)

        # final output layer
        # dense.shape = batch_size, n_categories
        output = self.dense_to_output_map(dense)

        # sort the output to match the original order
        output = output[orig_i, :]

        return nn.functional.log_softmax(output, dim=1)


class TopicCEncSimpleBPemb(TopicC):
    def __init__(self,
                 embed_size,
                 output_size,
                 enc_hidden_size):
        super(TopicCEncSimpleBPemb, self).__init__()
        print("init TopicCEncSimpleBPemb model...")

        self.embedding_model = BPEmb(dim=embed_size, lang="en", vs=100000)
        self.encoder = nn.GRU(
            input_size=embed_size,
            hidden_size=enc_hidden_size,
            num_layers=2,
            bidirectional=True
        )
        self.seq_to_output_map = nn.Linear(2 * enc_hidden_size, output_size, bias=False)

    def embed_sequence(self, sequence: str) -> torch.Tensor:
        v_ids = self.embedding_model.encode_ids(sequence)
        return torch.tensor(self.embedding_model.vectors[v_ids]).to(self._device)

    def create_seq_vecs(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # returns the padded seq vector, lengths and original order index
        # sequences are sorted by length, and can be reverted to their original
        # order with the unsorting index vector

        # start by embedding the sequence vectors
        seq_vecs = [self.embed_sequence(s) for s in sequences]

        # sort lengths and set the device
        lengths = torch.tensor([seq_v.shape[0] for seq_v in seq_vecs])
        lengths, sort_i = lengths.sort(descending=True)
        _, orig_i = sort_i.sort()

        # pad the seq vecs and sort by length (dim 2 is the batch dimension)
        seq_vec_pad = rnn.pad_sequence(seq_vecs).to(self._device)
        seq_vec_pad = seq_vec_pad[:, sort_i, :]

        return seq_vec_pad, lengths, orig_i

    def forward(self, sequences: List[str]) -> torch.Tensor:
        # Make the word embeddings for each sequence
        pad_seq_vecs, lengths, orig_i = self.create_seq_vecs(sequences)

        # pack the sequence for the GRU
        # packed_seq_vecs.shape = max_seq_len, batch_size, embedding_dim
        packed_seq_vecs = rnn.pack_padded_sequence(pad_seq_vecs, lengths)

        # run through the GRU
        _, h_n = self.encoder(packed_seq_vecs)
        seq_output = torch.cat((h_n[0], h_n[1]), dim=1)
        output = self.seq_to_output_map(seq_output)
        # sort the output to match the original order
        output = output[orig_i, :]

        return nn.functional.log_softmax(output, dim=1)
