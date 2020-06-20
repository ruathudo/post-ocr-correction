import os
import re
from functools import reduce
import numpy as np
from numpy import random

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from sklearn.model_selection import train_test_split


from . import models
from .utils import ids2text


def build_dictionary():
    letters = []

    with open('data/unicodes.txt') as f:
        for line in f:
            c = line.split()[1]
            letters.append(c)

    special_tokens = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<ctx>': 3, '<unk>': 4, '<sep>': 5}
    # special_tokens = {'<pad>':0, '<sos>':1, '<eos>':2, '<sep>':3, '<unk>':4}

    dictionary = Dictionary([letters])  # initialize a Dictionary
    dictionary.patch_with_special_tokens(special_tokens)

    return dictionary


def read_corpus(name, max_len=20, test_size=5000):
    filepath = 'yle-corpus/data/'

    with open(os.path.join(filepath, name), 'r') as f:
        # remove label and url from text
        text = f.read()

    text = strip_multiple_whitespaces(text)
    text = re.sub(r'__label__\S*\s', '', text)
    text = re.sub(r'\S?http\S+', '', text)
    text = text.lower()
    text = text.split()
    # dcm = [w for w in text if len(w) < max_len + 4 and len(w) > max_len]
    text = [w for w in text if len(w) <= max_len]
    # ml = max([len(w) for w in text])
    train, test = train_test_split(text, test_size=test_size, shuffle=False)

    return train, test
