import torch
import torch.nn as nn
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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