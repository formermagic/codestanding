import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=input_dim, embedding_dim=emb_dim
        )
        self.rnn = nn.GRU(
            input_size=emb_dim, hidden_size=enc_hid_dim, bidirectional=True
        )
        self.linear = nn.Linear(
            in_features=enc_hid_dim * 2, out_features=dec_hid_dim
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        outputs, hidden = self.rnn(embedded)

        hidden_cat_layers = torch.cat(
            (hidden[-2, :, :], hidden[-1, :, :]), dim=1
        )

        # Like in the original paper
        # hidden_cat_layers = torch.tanh(self.linear(hidden[-1, :, :]))

        hidden = torch.tanh(self.linear(hidden_cat_layers))

        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, enc hid dim]
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(
            in_features=(enc_hid_dim * 2) + dec_hid_dim,
            out_features=dec_hid_dim,
        )
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor):
        seq_len = encoder_outputs.size()[0]
        batch_size = encoder_outputs.size()[1]

        # [batch_size, seq_len, dec_hid_dim]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        # [batch_size, seq_len, enc_hid_dim * 2]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # [batch_size, seq_len, enc_hid_dim * 2 + dec_hid_dim]
        query_layers = torch.cat((hidden, encoder_outputs), dim=2)
        # [batch_size, seq_len, dec_hid_dim]
        energy = torch.tanh(self.attn(query_layers))

        # We need to get scores of size [batch_size, seq_len]
        # so we just multiply matrices with trycky permutations
        # to get rid of dec_hid_dim axis

        # [batch_size, 1, dec_hid_dim]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        # [batch_size, dec_hid_dim, seq_len]
        energy = energy.permute(0, 2, 1)

        # [batch_size, seq_len]
        scores = torch.bmm(v, energy).squeeze(1)

        # [batch_size, seq_len]
        return F.softmax(scores, dim=1)


class Decoder(nn.Module):
    def __init__(
        self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention
    ):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(
            num_embeddings=output_dim, embedding_dim=emb_dim
        )
        # encoder is bidirectional, so input_size=(enc_hid_dim * 2) + emb_dim
        self.rnn = nn.GRU(
            input_size=(enc_hid_dim * 2) + emb_dim, hidden_size=dec_hid_dim
        )
        self.out = nn.Linear(
            in_features=(enc_hid_dim * 2) + emb_dim + dec_hid_dim,
            out_features=output_dim,
        )

        self.droupout = nn.Dropout(dropout)

    def forward(self, x, hidden, encoder_outputs):
        # [1, batch_size]
        x = x.unsqueeze(0)
        # [1, batch_size, emb_dim]
        embedded = self.dropout(self.embedding(x))

        # [batch_size, seq_len]
        attn = self.attention(hidden, encoder_outputs)
        # [batch_size, 1, seq_len]
        attn = attn.unsqueeze(1)

        # [batch_size, seq_len, enc_hid_dim * 2]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # [batch_size, 1, enc_hid_dim * 2]
        weighted = torch.bmm(attn, encoder_outputs)
        # [1, batch_size, enc_hid_dim * 2]
        weighted = weighted.permute(1, 0, 2)

        # [1, batch_size, emb_dim + enc_hid_dim * 2]
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # [1, batch_size, dec_hid_dim]
        rnn_hidden = hidden.unsqueeze(0)

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]
        output, hidden = self.rnn(rnn_input, rnn_hidden)

        # [batch_size, emb_dim]
        embedded = embedded.squeeze(0)
        # [batch_size, dec_hid_dim]
        output = output.squeeze(0)
        # [batch_size, enc_hid_dim * 2]
        weighted = weighted.squeeze(0)

        # [batch_size, emb_dim + dec_hid_dim + enc_hid_dim * 2]
        output_cat = torch.cat((embedded, output, weighted), dim=1)
        # [batch_size, output_dim]
        pred = self.out(output_cat)

        return pred, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]

        batch_size = src.size()[1]
        trg_len = trg.size()[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)
        outputs = outputs.to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        x = trg[0, :]

        for t in range(1, trg_len):
            pred, hidden = self.decoder(x, hidden, encoder_outputs)
            outputs[t] = pred
            teacher_forcing = random.random() < teacher_forcing_ratio
            best_pred = pred.argmax(dim=1)
            x = trg[t] if teacher_forcing else best_pred

        return outputs


def train(
    model: Seq2Seq,
    iterator,
    optimizer: optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    clip,
):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.source
        trg = batch.target

        optimizer.zero_grad()

        # [trg len, batch size, output dim]
        output = model(src, trg)

        output_dim = output.shape[-1]

        # [trg len * batch size, output dim]
        output = output[1:].view(-1, output_dim)
        # [trg len * batch size, output dim]
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)
