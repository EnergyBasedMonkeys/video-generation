import torch
from torch import Tensor
from torch.nn import Transformer, Module, Dropout, Linear

import math

class PositionalEncoding(Module):
    def __init__( self, emb_size: int, dropout: float, maxlen: int = 11):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class Seq2SeqTransformer(Module):
    def __init__(
          self,
          num_encoder_layers: int,
          num_decoder_layers: int,
          emb_size: int,
          nhead: int,
          enc_dim: int,
          dim_feedforward: int = 512,
          dropout: float = 0.1
        ):
        super().__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.generator = Linear(emb_size, enc_dim)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor, tgt_mask: Tensor):
        src_emb = self.positional_encoding(src).permute(1, 0, 2)
        tgt_emb = self.positional_encoding(trg).permute(1, 0, 2)
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask, 
            tgt_mask,
            None,
            None, 
            None, 
            None
        )

        return self.generator(outs).permute(1, 0, 2)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(src), 
            src_mask
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
          self.positional_encoding(tgt), 
          memory,
          tgt_mask
        )
