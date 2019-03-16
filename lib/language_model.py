import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class WordEmbedding(nn.Module):
    """Word Embedding

    The n_tokens-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, n_tokens, emb_dim, dropout):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(n_tokens+1, emb_dim, padding_idx=n_tokens)
        self.dropout = nn.Dropout(dropout)
        self.n_tokens = n_tokens
        self.emb_dim = emb_dim

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.n_tokens, self.emb_dim)
        self.emb.weight.data[:self.n_tokens] = weight_init

    def freeze(self):
        self.emb.weight.requires_grad = False

    def defreeze(self):
        self.emb.weight.requires_grad = True

    def forward(self, x):
        emb = self.emb(x)
        emb = self.dropout(emb)
        return emb


class LanguageModel(nn.Module):

    def __init__(self, in_dim, hid_dim, n_layers, bidirectional, dropout, rnn_type='GRU'):

        super(LanguageModel, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU

        self.rnn = rnn_cls(
            in_dim, hid_dim, n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirectional)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.n_layers * self.ndirections, batch, self.hid_dim)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)

        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, :self.hid_dim]
        backward = output[:, 0, self.hid_dim:]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)
        return output
