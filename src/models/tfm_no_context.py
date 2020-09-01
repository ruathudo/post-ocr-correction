import pandas as pd
import numpy as np
from numpy import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm

from ..utils import create_noise, random_noise, get_correction


class SeqDataset(Dataset):

    def __init__(self, text, dictionary, rand_rate=0.5, window=1):
        self.text = text
        self.dictionary = dictionary
        self.rand_rate = rand_rate
        self.chars = list('abcdefghijklmnopqrstuvwxyzäåö')

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        # get a window of words
        trg_word = self.text[idx]
        src_word = self.text[idx]
        # add noise to the words and tokenize

        is_rand = False

        if len(src_word) < 3:
            is_rand = True
        elif random.rand() <= self.rand_rate or src_word[0] not in self.chars or src_word[-1] not in self.chars:
            src_word = random_noise(src_word)
            is_rand = True

        src_tokens = list(src_word)
        trg_tokens = list(trg_word)

        trg_ids = self.dictionary.doc2idx(trg_tokens, unknown_word_index=4)
        src_ids = self.dictionary.doc2idx(src_tokens, unknown_word_index=4)

        # add sos and eos token id
        trg_ids = [1] + trg_ids + [2]
        src_ids = [1] + src_ids + [2]

        trg_ids = torch.tensor(trg_ids, dtype=int)
        src_ids = torch.tensor(src_ids, dtype=int)

        return src_ids, trg_ids, is_rand


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
        src, trg, rands = zip(*batch)

        if self.rand_mode == 'mix' or self.rand_mode == 'trained':
            df_src = pd.DataFrame(list(zip(src, rands)), columns=["tokens", "is_rand"])
            non_errs = df_src.loc[df_src['is_rand'] == False]['tokens'].tolist()

            lens = torch.tensor([len(t) for t in non_errs])

            non_errs = pad_sequence(non_errs)
            noise_tokens = create_noise(non_errs, lens, self.noise_model, self.device)

            df_src.loc[df_src['is_rand'] == False, 'tokens'] = np.array(noise_tokens, dtype=object)

            src = df_src['tokens'].tolist()
            src = [torch.LongTensor(i) for i in src]

        # pad
        src = pad_sequence(src).permute(1, 0)
        trg = pad_sequence(trg).permute(1, 0)

        return src, trg


def train(model, data_loader, optimizer, criterion, device, scaler, mp=False):
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

        with autocast(enabled=mp):
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
            epoch_loss += loss.item()

        scaler.scale(loss).backward()
        # loss.backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        # optimizer.step()
        scaler.update()

    epoch_loss = epoch_loss / len(data_loader)
    acc = total_correct / total_sample

    return epoch_loss, acc


def evaluate(model, data_loader, criterion, device, mp=False):

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

            with autocast(enabled=mp):
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
