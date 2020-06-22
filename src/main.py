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
from .models.tfm_no_context import SeqDataset, Collator, train, evaluate
from .data_handler import build_dictionary, read_corpus
from . import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add training options')
    parser.add_argument('--data', required=True, help='Dataset Name')
    parser.add_argument('--model', required=True, help='Model Name')
    parser.add_argument('--update', action='store_const', const=True, default=False, help='Continue training')
    parser.add_argument('--batch', default=128, type=int, help='Batch Size')
    parser.add_argument('--epoch', default=1, type=int, help='Epoch Number')

    args = parser.parse_args()

    DATA = args.data
    MODEL_NAME = args.model
    IS_UPDATE = args.update
    BATCH_SIZE = args.batch
    N_EPOCHS = args.epoch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device('cpu')
    print('build dictionary')
    dictionary = build_dictionary()

    print('create data loader')
    train_data, test_data = read_corpus(DATA, max_len=30, test_size=50000)
    noise_model = error_gen.load_noise_model('models/error_generator_model.pt', device_cpu)
    train_set = SeqDataset(train_data, dictionary)
    test_set = SeqDataset(test_data, dictionary)

    collate_fn = Collator(noise_model, device_cpu)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, collate_fn=collate_fn, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, collate_fn=collate_fn, num_workers=4, drop_last=True)

    print('init model')
    criterion = nn.CrossEntropyLoss()
    # if update model, load the pretrained file
    pretrained_file = MODEL_NAME + '.pt' if IS_UPDATE else None
    model, optimizer = transfomer.init_model(dictionary, device, pretrained_file=pretrained_file)

    print('train and evaluate')
    for epoch in range(N_EPOCHS):
        t0 = time.time()

        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, test_loader, criterion, device)

        duration = (time.time() - t0) / 60

        log = f'Epoch: {epoch+1} | Time: {duration:.2f} m | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} | Train Acc: {train_acc:.4f} | Val. Acc: {valid_acc:.4f}'

        print('\n')
        print(log)
        print('\n')
        # write to log
        utils.write_log(log, MODEL_NAME + '.txt')
        # save checkpoint
        utils.save_model(model, optimizer, {}, MODEL_NAME + '.pt')
        # utils.upload_s3('tf_mix_noctx_full.pt', 'tf_mix_noctx_full.txt')
        # print('Uploaded model and log \n')
