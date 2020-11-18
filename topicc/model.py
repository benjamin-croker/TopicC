from typing import List

import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.utils.rnn as rnn

from bpemb import BPEmb


class TopicC(nn.Module):
    def __init__(self,
                 output_size,
                 enc_hidden_size,
                 attention_size,
                 dense_size,
                 dropout_rate=0.2,
                 # arguments for the bpemb model
                 dim=300, lang="en", vs=100000):
        super(TopicC, self).__init__()

        # todo: factor out params
        self.embedding_model = BPEmb(dim=dim, lang=lang, vs=vs)
        self.encoder = nn.LSTM(
            input_size=dim,
            hidden_size=enc_hidden_size,
            bidirectional=True
        )
        # 2* since lstm is bi-directional
        self.enc_to_att_map = nn.Linear(2 * enc_hidden_size, attention_size, bias=False)
        self.att_to_pointer_map = nn.Linear(attention_size, 1, bias=False)
        self.seq_to_dense_map = nn.Linear(4 * enc_hidden_size, dense_size, bias=False)
        self.dense_to_output_map = nn.Linear(dense_size, output_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def embed_sequence(self, sequence: str) -> torch.Tensor:
        v_ids = self.embedding_model.encode_ids(sequence)
        return torch.tensor(self.embedding_model.vectors[v_ids])

    @staticmethod
    def pack_seq_vecs(seq_vecs: List[torch.Tensor]) -> rnn.PackedSequence:
        # adds zero vectors as padding
        seq_vec_pad = rnn.pad_sequence(seq_vecs)
        # pack them suitable for an LSTM or other RNN
        return rnn.pack_padded_sequence(
            seq_vec_pad, lengths=[v.shape[0] for v in seq_vecs]
        )

    def forward(self, sequences: List[str]) -> torch.Tensor:
        # Make the word embeddings for each sequence
        # list of Tensors of dim seq_len, embed_size
        seq_vecs = [self.embed_sequence(s) for s in sequences]
        # sort in descending order of length
        seq_vecs = sorted(seq_vecs, key=len)[::-1]
        seq_last = [len(seq_vec)-1 for seq_vec in seq_vecs]
        # pack the sequence for the LSTM
        packed_seq_vecs = self.pack_seq_vecs(seq_vecs)

        # run through the LSTM
        enc_outputs, (h_c, _) = self.encoder(packed_seq_vecs)
        # unpack the sequence
        # enc_outputs.shape = max_seq_len, batch_size, 2*enc_hidden_size
        enc_outputs, _ = nn.utils.rnn.pad_packed_sequence(enc_outputs)

        # encoder masks to indicate which parts of the sequence should be considered
        enc_masks = torch.zeros(enc_outputs.shape[0], enc_outputs.shape[1], 1, dtype=torch.bool)
        for i, last in enumerate(seq_last):
            enc_masks[(last + 1):, i,  0] = True

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

        # combine with the hidden and cell states
        # get the index of the hidden state for the last part of each sequence
        seq_last = torch.tensor(seq_last).unsqueeze(-1).repeat(1, enc_outputs.shape[1]).unsqueeze(-1)
        enc_outputs = torch.gather(enc_outputs, dim=2, index=seq_last).squeeze()
        seq_output = torch.cat((enc_outputs, att_output), dim=1)

        # have a dense non-linear layer
        # dense.shape = batch_size, dense_size
        dense = self.seq_to_dense_map(seq_output)
        dense = torch.tanh(dense)
        # dense = self.dropout(dense)

        # final output layer
        # dense.shape = batch_size, n_categories
        output = self.dense_to_output_map(dense)

        return nn.functional.log_softmax(output, dim=1)

    def loss(self, sequences: List[str], target: torch.Tensor):
        pred_probs = self.forward(sequences)
        return nn.functional.nll_loss(pred_probs, target, reduction='mean')

    def predict(self, sequences: List[str]):
        _, pred = self.forward(sequences).topk(1)
        return pred.squeeze()
