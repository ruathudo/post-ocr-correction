import os
import re
import time
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import gensim
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import strip_multiple_whitespaces

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Using', device)
print()
print('Torch version', torch.__version__)
print()
print('Gensim version', gensim.__version__)

with open('log.txt', 'w') as f:
    f.write('Using ' + str(device) + '\n')
    f.write('Torch version ' + torch.__version__ + '\n')
    f.write('Gensim version ' + gensim.__version__ + '\n')
