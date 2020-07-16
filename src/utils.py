import os
import numpy as np
from numpy import random
import torch
from torch.optim import Adam
from torch.cuda.amp import GradScaler
import boto3


def get_correction(output, target):
    # output[output==2] = 0
    # target[target==2] = 0

    # print(target.shape)
    # print(output.shape)
    diff = torch.sum((output != target), axis=1)
    acc = torch.sum(diff == 0)

    return acc.item()


def ids2text(ids, dictionary):
    text = ''
    special_keys = [0, 1, 2, 4]

    for i in ids:
        if i not in special_keys:
            text = text + dictionary[i]

    text = text.replace("<sep>", " ")
    text = text.replace("<ctx>", " | ")

    return text


def print_result(src, trg, out, dictionary, n=5):
    ids = np.random.randint(src.shape[1], size=n)

    for i in ids:
        print(ids2text(src[:, i], dictionary))
        print(ids2text(trg[:, i], dictionary))
        print(ids2text(out[:, i], dictionary))
        print("-" * 30)


def save_model(model, optimizer, scaler, model_name, epoch):

    filepath = os.path.join('models', model_name)

    if os.path.isfile(filepath + '.pt'):
        os.rename(filepath + '.pt', filepath + '_' + str(epoch - 1) + '.pt')

    torch.save({
        'epoch': epoch,
        'scaler': scaler.state_dict(),
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filepath + '.pt')


def load_model(model_name, model, device):
    filepath = os.path.join('models', model_name + '.pt')
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint['model'])
    model.eval()

    optimizer = Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])

    scaler = GradScaler(enabled=False)
    scaler.load_state_dict(checkpoint['scaler'])

    epoch = checkpoint['epoch'] or 0
    return model, optimizer, scaler, epoch


def write_log(line, filename):
    if not os.path.isdir('logs'):
        os.mkdir('logs')

    filepath = os.path.join('logs', filename)

    with open(filepath, 'a') as f:
        f.write(line + '\n')


def random_noise(text, noise_rate=0.07):
    chars = list('abcdefghijklmnopqrstuvwxyzäåö')
    rate = noise_rate
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


def create_noise(token_ids, lengths, model, device, noise_rate=0.07):
    # add sos and eos token
    token_ids = token_ids.to(device)
    lengths = lengths.to(device)

    # get the tensor shape (seq_len, batch)
    noise_words = []
    special_keys = range(6)

    # get encoded output
    with torch.no_grad():
        output = model(token_ids, token_ids, lengths, 0)
        output = torch.topk(output[1:], k=5, dim=2)[1]

        for i in range(output.shape[1]):
            word = output[:, i]  # (seqs, 1, 5)
            noise_tokens = []

            for c in word:
                if c[0] in special_keys:
                    break

                if np.random.rand() < noise_rate:
                    x = np.random.choice(5)
                else:
                    # otherwise get the highest output
                    x = 0

                noise_tokens.append(c[x].item())

            # if the model can't generate a good noise, then return the origin
            if len(noise_tokens) == 0:
                noise_tokens = token_ids[:, i].tolist()
            else:
                noise_tokens = [1] + noise_tokens + [2]
            noise_words.append(noise_tokens)

    return noise_words


def upload_s3(model_file, log_file):
    # upload logs
    bucket = 'aihub-files-dev'
    s3 = boto3.client('s3')
    log_path = os.path.join('logs', log_file)
    model_path = os.path.join('models', model_file)

    s3.upload_file(log_path, bucket, log_path)
    s3.upload_file(model_path, bucket, model_path)
