import numpy as np
from numpy import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from functools import reduce
from . import error_gen
from .. import utils


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):

        # src = [src len, batch size]
        # src_len = [batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        packed_embedded = pack_padded_sequence(embedded, src_len, enforce_sorted=False)

        packed_outputs, hidden = self.rnn(packed_embedded)

        # packed_outputs is a packed sequence containing all hidden states
        # hidden is now from the final non-padded element in the batch

        outputs, _ = pad_packed_sequence(packed_outputs)

        # outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros

        # outputs = [src len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):

        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        # batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention = [batch size, src len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.fc_out = nn.Linear(
            (enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):

        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        # mask = [batch size, src len]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs, mask)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0), a.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_len, trg, mask, teacher_forcing_ratio=0.5):

        # src = [src len, batch size]
        # src_len = [batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        outputs[0, :, 1] = 1

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        # mask = [batch size, src len]

        for t in range(1, trg_len):

            # insert input token embedding, previous hidden state, all encoder hidden states
            #  and mask
            # receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = np.random.rand() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


class SeqDataset(Dataset):

    def __init__(self, text, dictionary, window=3, err_rate=0.3, noise_rate=0.1, noise='rand'):
        self.text = text
        self.dictionary = dictionary
        self.window = window
        self.err_rate = err_rate
        self.noise_rate = noise_rate
        self.length = len(text) - window + 1
        self.noise = noise

        # load trained noise model
        if noise != 'rand':
            self.noise_model = self.load_noise_model()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # get a window of words
        text_window = self.text[idx: idx + self.window]
        # add noise to the words and tokenize
        mid_index = int(self.window / 2)
        mid_token = text_window[mid_index]
        trg_tokens = list(mid_token)

        noise_token = self.add_noise(mid_token)
        text_window[mid_index] = noise_token
        src_tokens = self.tokenizer(text_window)

        # print(trg_tokens, src_tokens)

        src_ids = self.dictionary.doc2idx(src_tokens, unknown_word_index=4)
        trg_ids = self.dictionary.doc2idx(trg_tokens, unknown_word_index=4)

        # add sos and eos token id
        src_ids = [1] + src_ids + [2]
        trg_ids = [1] + trg_ids + [2]

        src_ids = torch.tensor(src_ids, dtype=int)
        trg_ids = torch.tensor(trg_ids, dtype=int)

        return src_ids, trg_ids

    def add_noise(self, text):
        if len(text) > 2:
            if self.noise == 'rand':
                text = self.random_noise(text)
            else:
                text = self.trained_noise(text)

        return text

    def tokenizer(self, text_window):
        tokens = [list(x) for x in text_window]
        tokens = reduce(lambda a, b: a + ['<sep>'] + b, tokens)
        return tokens

    def random_noise(self, text):
        chars = list('abcdefghijklmnopqrstuvwxyzäåö')
        rate = self.noise_rate
        str_len = len(text)

        if random.rand() < rate * str_len:
            # Replace a character with a random character
            pos = random.randint(len(text))
            # only replace for alphabet
            if text[pos] in chars:
                text = text[:pos] + random.choice(chars[:-1]) + text[pos + 1:]

        if random.rand() < rate * str_len:
            # Delete a character
            pos = random.randint(len(text))
            # only for alphabet
            if text[pos] in chars:
                text = text[:pos] + text[pos + 1:]

        if random.rand() < rate * str_len:
            # Add a random character
            pos = random.randint(len(text))
            if text[pos] in chars:
                text = text[:pos] + random.choice(chars[:-1]) + text[pos:]

        return text

    def load_noise_model(self):
        device = torch.device('cpu')
        check_point = torch.load(self.noise)

        encoder = error_gen.Encoder(*check_point['args']).to(device)
        decoder = error_gen.Decoder(*check_point['args']).to(device)

        model = error_gen.Seq2Seq(encoder, decoder, device).to(device)
        model.load_state_dict(check_point['state_dict'])
        model.eval()

        return model

    def trained_noise(self, text):
        model = self.noise_model
        rate = self.noise_rate
        device = torch.device('cpu')

        tokens = list(text)
        src_ids = self.dictionary.doc2idx(tokens, unknown_word_index=4)
        # add sos and eos token
        src_ids = [1] + src_ids + [2]
        src_ids = torch.LongTensor(src_ids).to(device)

        # get the tensor shape (seq_len, batch)
        src_ids = src_ids.unsqueeze(1)
        src_len = torch.LongTensor([len(src_ids)]).to(device)

        # get encoded output
        with torch.no_grad():
            context = model.encoder(src_ids, src_len)

        # max len for target will not more than 10 characters from src
        max_len = len(src_ids) + 3
        # init trg indices with <sos>
        trg_ids = [1]

        with torch.no_grad():
            hidden = context

            for t in range(max_len):
                # feed last token of target to decoder
                trg_token = torch.LongTensor([trg_ids[-1]]).to(device)

                output, hidden = model.decoder(trg_token, hidden, context)

                topk = torch.topk(output, k=5, dim=1)[1]
                # only apply rand in topk for n% letters
                if np.random.rand() < rate:
                    x = np.random.choice(5, size=1)
                    pred_token = topk[0][x].item()
                else:
                    # otherwise get the highest output
                    pred_token = topk[0][0].item()

                trg_ids.append(pred_token)
                # 2 is the index of eos
                if pred_token == 2:
                    break

        result = utils.ids2text(trg_ids, self.dictionary)
        return result


def collate_fn_pad(batch):
    src, trg = zip(*batch)
    # get sequence lengths
    lengths = torch.tensor([t.shape[0] for t in src])
    # trg_len = torch.tensor([ t.shape[0] for t in trg ])

    # pad
    src = pad_sequence(src)
    trg = pad_sequence(trg)
    # compute mask
    masks = (src != 0).permute(1, 0)
    return src, trg, lengths, masks


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def train(model, data_loader, optimizer, criterion, clip, device):

    model.train()

    epoch_loss = 0
    total_correct = 0
    total_sample = 0

    for batch in data_loader:

        src, trg, lens, masks = batch
        src = src.to(device)
        trg = trg.to(device)
        lens = lens.to(device)
        masks = masks.to(device)

        output = model(src, lens, trg, masks)

        y_pred = torch.argmax(output, 2)
        y_true = trg.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        total_sample += y_true.shape[1]
        total_correct += utils.get_correction(y_pred, y_true)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        optimizer.zero_grad()
        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(data_loader)
    acc = total_correct / total_sample

    return epoch_loss, acc


def evaluate(model, data_loader, criterion, device):

    model.eval()

    epoch_loss = 0
    total_correct = 0
    total_sample = 0

    with torch.no_grad():

        for batch in data_loader:

            src, trg, lens, masks = batch
            src = src.to(device)
            trg = trg.to(device)
            lens = lens.to(device)
            masks = masks.to(device)

            output = model(src, lens, trg, masks, 0)

            y_pred = torch.argmax(output, 2)
            y_pred = y_pred.detach().cpu().numpy()
            y_true = trg.detach().cpu().numpy()

            total_sample += y_true.shape[1]
            total_correct += utils.get_correction(y_pred, y_true)

            # output = model(src, src_len, trg, 0) #turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(data_loader)
    acc = total_correct / total_sample

    return epoch_loss, acc


def predict(data_loader, model, device):
    model.eval()

    with torch.no_grad():

        for batch in data_loader:

            src, trg, lens, masks = batch
            src = src.to(device)
            trg = trg.to(device)
            lens = lens.to(device)
            masks = masks.to(device)

            output = model(src, lens, trg, masks, 0)
            output = torch.argmax(output, 2)

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]
            src = src.detach().cpu().numpy()
            trg = trg.detach().cpu().numpy()
            output = output.detach().cpu().numpy()

            yield src, trg, output
