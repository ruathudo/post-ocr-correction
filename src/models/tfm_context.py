import pandas as pd
import numpy as np
from numpy import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
from tqdm import tqdm

from functools import reduce
from ..utils import create_noise, random_noise, get_correction


class SeqDataset(Dataset):

    def __init__(self, text, dictionary, window=3, rand_rate=0.5):
        self.text = text
        self.dictionary = dictionary
        self.window = window
        self.rand_rate = rand_rate
        self.length = len(text) - window + 1
        self.chars = list('abcdefghijklmnopqrstuvwxyzäåö')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # get a window of words
        text_window = self.text[idx: idx + self.window]
        # add noise to the words and tokenize
        mid_index = int(self.window / 2)
        mid_word = text_window[mid_index]

        err_word = mid_word
        is_rand = False

        if len(mid_word) < 3:
            is_rand = True
        elif random.rand() <= self.rand_rate or mid_word[0] not in self.chars or mid_word[-1] not in self.chars:
            err_word = random_noise(mid_word)
            is_rand = True

        # because context having the form:
        # sos + l1 + sep + l2 + ctx + mid_word + ctx + r1 + sep + r2 + eos + pad
        l_ctx = text_window[:mid_index]
        r_ctx = text_window[mid_index + 1:]

        l_tokens = self.tokenizer(l_ctx)
        r_tokens = self.tokenizer(r_ctx)
        src_tokens = list(err_word)
        trg_tokens = list(mid_word)

        # print(trg_tokens, src_tokens)

        l_ctx_ids = self.dictionary.doc2idx(l_tokens, unknown_word_index=4)
        r_ctx_ids = self.dictionary.doc2idx(r_tokens, unknown_word_index=4)
        trg_ids = self.dictionary.doc2idx(trg_tokens, unknown_word_index=4)
        src_ids = self.dictionary.doc2idx(src_tokens, unknown_word_index=4)

        # add sos and eos token id
        l_ctx_ids = [1] + l_ctx_ids + [3]
        r_ctx_ids = [3] + r_ctx_ids + [2]
        trg_ids = [1] + trg_ids + [2]
        src_ids = [1] + src_ids + [2]

        l_ctx_ids = torch.tensor(l_ctx_ids, dtype=int)
        r_ctx_ids = torch.tensor(r_ctx_ids, dtype=int)
        trg_ids = torch.tensor(trg_ids, dtype=int)
        src_ids = torch.tensor(src_ids, dtype=int)

        return l_ctx_ids, r_ctx_ids, trg_ids, src_ids, is_rand

    def tokenizer(self, text_window):
        tokens = [list(x) for x in text_window]
        tokens = reduce(lambda a, b: a + ['<sep>'] + b, tokens)
        return tokens


class Collator(object):
    def __init__(self, noise_model, device, rand_mode="mix"):
        """
        rand_mode: mix, rand, trained
        TODO implement rand_mode none
        """
        self.noise_model = noise_model
        self.device = device
        self.rand_mode = rand_mode

    def __call__(self, batch):
        l_ctx, r_ctx, trg_ids, src_ids, rands = zip(*batch)
        # get sequence lengths
        # lengths = torch.tensor([ t.shape[0] for t in src ])
        if self.rand_mode == "mix":
            df_src = pd.DataFrame(list(zip(src_ids, rands)), columns=['tokens', 'is_rand'])
            non_errs = df_src.loc[df_src['is_rand'] == False]['tokens'].tolist()
            lens = torch.tensor([len(t) for t in non_errs])
            non_errs = pad_sequence(non_errs)

            noise_tokens = create_noise(non_errs, lens, self.noise_model, self.device)

            df_src.loc[df_src['is_rand'] == False, 'tokens'] = noise_tokens

            src_tokens = df_src['tokens'].tolist()
            src_tokens = [torch.LongTensor(i[1:-1]) for i in src_tokens]

        elif self.rand_mode == "trained":
            lens = torch.tensor([t.shape[0] for t in src_ids])
            non_errs = pad_sequence(src_ids)
            noise_tokens = create_noise(non_errs, lens, self.noise_model, self.device)
            src_tokens = [torch.LongTensor(i[1:-1]) for i in noise_tokens]
        else:
            # remove sos and eos
            src_tokens = [i[1:-1] for i in src_ids]

        src = [torch.cat((x, y, z)) for x, y, z in zip(l_ctx, src_tokens, r_ctx)]

        # pad
        src = pad_sequence(src).permute(1, 0)
        trg = pad_sequence(trg_ids).permute(1, 0)

        return src, trg


def train(model, data_loader, optimizer, criterion, device, scaler):
    clip = 1
    model.train()
    epoch_loss = 0
    total_correct = 0
    total_sample = 0

    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc='Train'):
        optimizer.zero_grad()
        src, trg = batch
        src = src.to(device, non_blocking=True)
        trg = trg.to(device, non_blocking=True)
        with autocast(enabled=False):
            output, _ = model(src, trg[:, :-1])

            y_pred = torch.argmax(output, 2)
            # y_pred = y_pred.cpu().numpy()
            y_true = trg[:, 1:]

            total_sample += y_true.shape[0]
            total_correct += get_correction(y_pred, y_true)

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]
            loss = criterion(output, trg)

        scaler.scale(loss).backward()
        # loss.backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        # optimizer.step()
        scaler.update()

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
            src, trg = batch
            # src, trg = process_input(deepcopy(l_ctx), deepcopy(trg), deepcopy(r_ctx), noise_model)
            src = src.to(device, non_blocking=True)
            trg = trg.to(device, non_blocking=True)

            with autocast(enabled=False):
                output, _ = model(src, trg[:, :-1])

                y_pred = torch.argmax(output, 2)
                # y_pred = y_pred.cpu().numpy()
                y_true = trg[:, 1:]

                total_sample += y_true.shape[0]
                total_correct += get_correction(y_pred, y_true)

                # output = model(src, src_len, trg, 0) #turn off teacher forcing

                # trg = [trg len, batch size]
                # output = [trg len, batch size, output dim]

                output_dim = output.shape[-1]

                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)

                # trg = [(trg len - 1) * batch size]
                # output = [(trg len - 1) * batch size, output dim]

                loss = criterion(output, trg)

            epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(data_loader)
    acc = total_correct / total_sample

    return epoch_loss, acc
