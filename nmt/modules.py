import random

import torch
from torch import nn


def _softmax(x, c: int = 10):
    """
    Custom softmax method with possibility to set temperature
    :param x: input
    :param c: temperature
    """
    e_x = torch.exp(x / c)
    return e_x / torch.sum(e_x, dim=1).unsqueeze(1)


class Encoder(nn.Module):
    def __init__(
            self, input_size, emb_size, hidden_size, num_layers=2,
            dropout=0.1, is_bidirectional=False
    ):
        super().__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, emb_size)
        self.rnn = nn.LSTM(
            emb_size, hidden_size, num_layers=num_layers,
            bidirectional=is_bidirectional
        )
        self.out = nn.Linear((1+is_bidirectional) * hidden_size, hidden_size)
        self.bidirectional = is_bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        :param: src sentences (src_len x batch_size)
        """
        # embedded = (src_len x batch_size x embd_dim)
        embedded = self.embedding(src)
        # dropout over embedding
        embedded = self.dropout(embedded)
        outputs, (hidden, cell) = self.rnn(embedded)
        if self.bidirectional:
            hidden = torch.tanh(self.out(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
            hidden = hidden.unsqueeze(0)
        # [Attention return is for lstm, but you can also use gru]
        return hidden, outputs


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, is_bidirectional, temperature=1):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.temp = temperature

        self.attn = nn.Linear((1+is_bidirectional) * enc_hid_dim + dec_hid_dim, enc_hid_dim)
        self.v = nn.Linear(enc_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):

        # encoder_outputs = [src sent len, batch size, enc_hid_dim]
        # hidden = [1, batch size, dec_hid_dim]
        sent_len = encoder_outputs.shape[0]
        # repeat hidden and concatenate it with encoder_outputs
        hidden = hidden.permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src sent len, enc_hid_dim]
        # hidden = [batch size, 1, dec_hid_dim]
        hidden = hidden.repeat(1, sent_len, 1)
        hidden = torch.cat((hidden, encoder_outputs), dim=2)

        # calculate energy
        energy = torch.tanh(self.attn(hidden))
        # get attention, use softmax function which is defined, can change temperature
        attention = self.v(energy).squeeze(2)

        return _softmax(attention, self.temp)


class Decoder(nn.Module):
    def __init__(
            self, output_size, emb_size, hidden_size, attention,
            num_layer=2, dropout=0.1, is_bidirectional=False):
        super().__init__()

        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layer
        self.dropout = dropout
        self.bidirectional = is_bidirectional

        self.embedding = nn.Embedding(output_size, emb_size)

        self.rnn = nn.GRU(
            (1+is_bidirectional) * hidden_size + emb_size, hidden_size,
            num_layers=num_layer,
        )
        # (lstm embd, hid, layers, dropout)

        # Projection :hid_dim x output_dim
        self.out = nn.Linear(
            (1+is_bidirectional) * hidden_size+emb_size+hidden_size, output_size
        )
        self.dropout = nn.Dropout(dropout)
        self.attention = attention

    def forward(self, input_, hidden, outputs):
        # (1x batch_size)
        input_ = input_.unsqueeze(0)

        # (1 x batch_size x emb_dim)
        # embed over input and dropout
        embedded = self.embedding(input_)
        embedded = self.dropout(embedded)

        # get weighted sum of encoder_outputs
        att = self.attention(hidden, outputs).unsqueeze(1)
        encoder_outputs = outputs.permute(1, 0, 2)
        att = torch.bmm(att, encoder_outputs)
        att = att.permute(1, 0, 2)

        # concatenate weighted sum and embedded, break through the GRU
        # output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        output, hidden = self.rnn(
            torch.cat((embedded, att), dim=2), hidden
        )

        # sent len and n directions will always be 1 in the decoder

        # (batch_size x output_dim)
        # project out of the rnn on the output dim
        prediction = self.out(
            torch.cat((embedded.squeeze(0), att.squeeze(0), output.squeeze(0)), dim=1)
        )

        return prediction, hidden[-1, :, :].unsqueeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        # Hidden dimensions of encoder and decoder must be equal
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        :param: src (src_len x batch_size)
        :param: tgt
        :param: teacher_forcing_ration : if 0.5 then every second token is the ground truth input
        """

        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_size

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, enc_states = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input_ = trg[0]

        for t in range(1, max_len):
            output, hidden = self.decoder(input_, hidden, enc_states)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input_ = (trg[t] if teacher_force else top1)

        return outputs
