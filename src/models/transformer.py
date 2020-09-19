import os
import math
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from ..utils import load_model


class TransformerModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, n_head, hid_dim, n_layer, dropout=0.5, pad_idx=0):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pad_idx = pad_idx

        self.pos_encoder = PositionalEncoding(embed_dim, dropout)

        encoder_layers = TransformerEncoderLayer(embed_dim, n_head, hid_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layer)

        decoder_layers = TransformerDecoderLayer(embed_dim, n_head, hid_dim, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, n_layer)

        self.embed_src = nn.Embedding(vocab_size, embed_dim)
        self.embed_tgt = nn.Embedding(vocab_size, embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

        self.embed_dim = embed_dim

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _generate_pad_mask(self, seq):
        # seq = [batch size, seq len]
        mask = (seq == self.pad_idx).permute(1, 0)
        return mask

    def encoder(self, src):
        src_pad_mask = self._generate_pad_mask(src)
        src = self.embed_src(src) * math.sqrt(self.embed_dim)
        src = self.pos_encoder(src)
        encoded = self.transformer_encoder(src, src_key_padding_mask=src_pad_mask)

        return encoded, src_pad_mask

    def decoder(self, tgt, encoded, src_pad_mask):
        mem_pad_mask = src_pad_mask.clone()
        tgt_pad_mask = self._generate_pad_mask(tgt)

        device = tgt.device
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(0)).to(device)
        tgt = self.embed_tgt(tgt) * math.sqrt(self.embed_dim)
        tgt = self.pos_encoder(tgt)
        decoded = self.transformer_decoder(tgt, encoded, tgt_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=mem_pad_mask)
        output = self.fc_out(decoded)
        return output

    def forward(self, src, tgt):
        encoded, src_pad_mask = self.encoder(src)
        output = self.decoder(tgt, encoded, src_pad_mask)
        # print('src', src.shape)
        # print('tgt', tgt.shape)

        # print('src', src.shape)
        # print('tgt', tgt.shape)
        # print('src_pad', src_pad_mask.shape)
        # print('tgt_pad', tgt_pad_mask.shape)
        # print('tgt_mask', tgt_mask.shape)

        # print('encoded', encoded.shape)
        # print('hehe', torch.isnan(encoded).sum())

        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=150):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def initialize_weights(self):
    if hasattr(self, 'weight') and self.weight.dim() > 1:
        nn.init.xavier_uniform_(self.weight.data)


def init_model(dictionary, device, pretrained_file=None, mp=False):

    VOCAB_SIZE = len(dictionary)
    EMBED_DIM = 256
    LAYERS = 3
    HEADS = 8
    HID_DIM = 512
    DROPOUT = 0.1
    PAD_IDX = 0
    LEARNING_RATE = 0.0005

    model = TransformerModel(VOCAB_SIZE, EMBED_DIM, HEADS, HID_DIM, LAYERS, DROPOUT, PAD_IDX).to(device)

    if pretrained_file:
        # model_path = os.path.join('models', pretrained_file)
        model, optimizer, scaler, epoch = load_model(pretrained_file, model, device, mp)
    else:
        # lr=0.0005
        model.apply(initialize_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scaler = GradScaler(enabled=mp)
        epoch = 0

    return model, optimizer, scaler, epoch
