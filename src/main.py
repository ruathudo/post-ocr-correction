import os
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from multiprocessing import cpu_count
import tqdm

from .models import transfomer, error_gen
# from .models.tfm_no_context import SeqDataset, Collator, train, evaluate
from .models import tfm_context as ctx
from .models import tfm_no_context as nctx
from .data_handler import build_dictionary, read_corpus
from . import utils

import warnings
# Cause all warnings to raise exceptions:
# warnings.filterwarnings('error')
# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add training options')
    parser.add_argument('--mp', action='store_const', const=True, default=False, help='Mixed precision')
    parser.add_argument('--data', required=True, help='Dataset Name')
    parser.add_argument('--model', required=True, help='Model Name')
    parser.add_argument('--resume', action='store_const', const=True, default=False, help='Resume training')
    parser.add_argument('--batch', default=128, type=int, help='Batch Size')
    parser.add_argument('--epoch', default=1, type=int, help='Epoch Number')
    parser.add_argument('--rand', default=0.5, type=float, help='Random rate from 0-1')
    parser.add_argument('--window', default=3, type=int, help='window size')

    args = parser.parse_args()
    module = ctx if args.window > 1 else nctx

    MP = args.mp
    DATA = args.data
    MODEL_NAME = args.model
    RESUME = args.resume
    BATCH_SIZE = args.batch
    N_EPOCHS = args.epoch
    RAND_RATE = args.rand
    WINDOW = args.window
    RAND_MODE = 'mix'

    if RAND_RATE == 1.0:
        RAND_MODE = 'rand'
    elif RAND_RATE == 0.0:
        RAND_MODE = 'trained'

    print("Using rand mode:", RAND_MODE)
    print("Window size:", WINDOW)
    print("Using AMP mode:", MP)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device('cpu')
    print('Build dictionary')
    dictionary = build_dictionary()

    print('Create data loader')
    train_data, test_data = read_corpus(DATA, max_len=30, test_size=50000)
    noise_model = error_gen.load_noise_model('models/error_generator_model.pt', device_cpu)

    train_set = module.SeqDataset(train_data, dictionary, rand_rate=RAND_RATE, window=WINDOW)
    test_set = module.SeqDataset(test_data, dictionary, rand_rate=RAND_RATE, window=WINDOW)

    collate_fn = module.Collator(noise_model, device_cpu, rand_mode=RAND_MODE)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=True, collate_fn=collate_fn, num_workers=7, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             pin_memory=True, collate_fn=collate_fn, num_workers=7, drop_last=True)

    print('Init model')
    criterion = nn.CrossEntropyLoss()
    # if update model, load the pretrained file
    pretrained_file = MODEL_NAME if RESUME else None
    model, optimizer, scaler, epoch = transfomer.init_model(dictionary, device, pretrained_file, MP)

    print('Train and evaluate')
    for e in range(1, N_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = module.train(model, train_loader, optimizer, criterion, device, scaler, MP)
        valid_loss, valid_acc = module.evaluate(model, test_loader, criterion, device, MP)

        duration = (time.time() - t0) / 60

        log = f'Epoch: {epoch + e} | Time: {duration:.2f} m | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} | Train Acc: {train_acc:.4f} | Val. Acc: {valid_acc:.4f}'

        print('\n')
        print(log)
        print('\n')
        # write to log
        utils.write_log(log, MODEL_NAME + '.txt')
        # save checkpoint
        utils.save_model(model, optimizer, scaler, MODEL_NAME, epoch + e)
        # utils.upload_s3('tf_mix_noctx_full.pt', 'tf_mix_noctx_full.txt')
        # print('Uploaded model and log \n')
