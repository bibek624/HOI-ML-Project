import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import copy
from transformers import CLIPTextModel, CLIPTokenizer
# Load pre-trained CLIP text encoder and tokenizer
clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")



class HOIDecoder(nn.Module):
    def __init__(self, embed_size = 256, num_heads=8, num_layers = 5, ff_hid_dim = 2000,dropout=0.1):
        super(HOIDecoder, self).__init__()
        
        # decoder_layer = TransformerDecoderLayer(embed_size, num_heads, ff_hid_dim,
        #                                         dropout, 'relu', False)
        # decoder_norm = nn.LayerNorm(embed_size)
        # self.decoder = TransformerDecoder(decoder_layer, num_layers, decoder_norm,
        #                                   return_intermediate=True)

        
        # Define the decoder layer
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, 
            nhead=num_heads, 
            dim_feedforward=ff_hid_dim,
            dropout=dropout,
            batch_first=True
        )

        
        
        # Create the TransformerDecoder with multiple layers
        self.decoder = nn.TransformerDecoder(
            self.decoder_layer, 
            num_layers=num_layers,
        )
        
    
    def forward(self, Fe, Qr):
        """
        Forward pass for the HOI Decoder
        
        Args:
            Fe (torch.Tensor): Encoder output (B, T, D) -> Features from the encoder.
            p (torch.Tensor): Positional embeddings (B, T, D)
            Qr (torch.Tensor): Refined queries (B, Nq, D) -> Refined queries for decoder
        
        Returns:
            
        """
        
        
    
        # Apply the Transformer Decoder
        output = self.decoder(Qr, Fe)  # (B,Nq, D) output
        
 
        return output


