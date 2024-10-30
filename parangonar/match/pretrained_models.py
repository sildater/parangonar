#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains pretrained pytorch models.
checkpoints in assets. 
loaded in matchers and online_matchers.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.normalization import LayerNorm
import torch.nn.functional as F

# ALIGNMENT TRANSFORMER

class PositionalEncoding(nn.Module):
    def __init__(self, 
                 dim_model, 
                 dropout_p, 
                 max_len):
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) 
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-np.log(10000.0)) / dim_model) 
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:

        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class AlignmentTransformer(nn.Module):
    """
    """
    # Constructor
    def __init__(
        self,
        token_number = 91,
        dim_model = 128,
        dim_class = 2,
        num_heads = 4,
        num_decoder_layers = 6,
        dropout_p = 0.1,
    ):
        super().__init__()
        
        self.tokennumber = token_number
        self.dim_model = dim_model
        self.dim_class = dim_class
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=self.dim_model , dropout_p=dropout_p, max_len=50000
        )
        self.embedding = nn.Embedding(self.tokennumber,self.dim_model ) 
        
        # DECODER LAYERS
        D_layers = TransformerEncoderLayer(self.dim_model, 
                                            nhead = self.num_heads, 
                                            dim_feedforward = self.dim_model, 
                                            dropout=dropout_p)
        
        self.transformerDECODER = TransformerEncoder(
            encoder_layer = D_layers,
            num_layers = num_decoder_layers,
        )
        self.out = nn.Linear(self.dim_model, self.dim_class)
        
    def forward(self, src, tgt_mask=None, tgt_pad_mask=None):
        src = self.embedding(src)
        src = torch.sum(src, dim=-2)
        src = src.permute(1,0,2)
        src = self.positional_encoder(src)
        transformer_out = self.transformerDECODER(src=src, 
                                                  mask=tgt_mask, 
                                                  src_key_padding_mask = tgt_pad_mask)
        out = self.out(transformer_out)
        
        return out
      
    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.zeros(size, size)
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int = -1) -> torch.tensor:
        return (matrix == pad_token)
    
# TheGlueNote

class TheGlueNote(nn.Module):
    """
    GlueNote non-causal transformer encoder
    with learned positional encoding
    """
    # Constructor
    def __init__(
        self,
        device,
        token_number = 314,
        position_number = 512,
        dim_model = 128,
        dim_feedforward = None,
        num_heads = 8,
        num_decoder_layers = 4,
        dropout_p = 0.0,
        activation = nn.GELU(),
        using_decoder = True
    ):
        super().__init__()
        self.device = device
        self.token_number = token_number
        self.position_number = position_number + 1 
        self.dim_model = dim_model
        self.activation = activation
        if dim_feedforward is not None:
            self.dim_feedforward = dim_feedforward
        else: 
            self.dim_feedforward = self.dim_model * 4
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.dropout_p = dropout_p
        self.using_decoder = using_decoder
        # LAYERS
        self.positions = torch.arange(self.position_number * 2).to(self.device)
        self.positional_encoder = nn.Embedding(self.position_number * 2,self.dim_model) 
        self.embedding = nn.Embedding(self.token_number,self.dim_model) 
        
        # DECODER LAYERS
        D_layers = TransformerEncoderLayer(self.dim_model, 
                                            nhead = self.num_heads, 
                                            dim_feedforward = self.dim_feedforward, 
                                            dropout=self.dropout_p,
                                            norm_first = True,
                                            activation = self.activation)

        self.layer_normalization= LayerNorm(self.dim_model)
        self.layer_normalization_input = LayerNorm(self.dim_model)

        self.transformerDECODER = TransformerEncoder(
            encoder_layer = D_layers,
            num_layers = num_decoder_layers,
            norm = self.layer_normalization,
            enable_nested_tensor=False
        )

        self.mlp1 = nn.Linear(self.dim_model,self.dim_feedforward)
        self.mlp_activation = activation = self.activation
        self.mlp2 = nn.Linear(self.dim_feedforward,self.dim_model)

        self.embed_out = nn.Linear(self.dim_model,self.dim_model)
        
    def forward(self, src, return_confidence_matrix = False):
        # # let's just only use pitch for now, no aggregation
        # src = src[:,1::4]
        src = self.embedding(src)
        
        # AGGREGATING 4 ATTRIBUTES
        eshape = src.shape
        src = torch.sum(src.reshape(eshape[0], -1, 4, eshape[-1]), dim=-2)
        
        # to obtain size (sequence length, batch_size, dim_model)
        src = src.permute(1,0,2)
        # POSITIONAL ENCODING
        pos = self.positional_encoder(self.positions)
        src += pos.unsqueeze(1)
        # src = self.positional_encoder(src)
        src = self.layer_normalization_input(src)
        # Transformer blocks - Out size = (sequence length, batch_size, dim_model)
        transformer_out = self.transformerDECODER(src=src)    
        # mlp_out = self.mlp2(self.melp_activation(self.mlp1(transformer_out))) 
        mlp_out = self.embed_out(transformer_out)
        predictions = torch.einsum('ibk,jbk->bij', mlp_out[:self.position_number,:,:], mlp_out[self.position_number:,:,:])#

        if return_confidence_matrix:
            conf_matrix = F.softmax(predictions, 1) * F.softmax(predictions, 2)
            return conf_matrix
        else:    
            if self.using_decoder:
                return predictions, mlp_out
            else:
                return predictions