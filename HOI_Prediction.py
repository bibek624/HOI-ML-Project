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



class HOIPrediction(nn.Module):
    def __init__(self, num_act_classes, num_obj_classes, embed_size = 256,  ff_hid_dim = 2000,  dropout=0.1):
        super(HOIPrediction, self).__init__()
        
        
        # Feedforward layers for predictions (4 small FFNs)
        self.ffn_bh = nn.Sequential(
            nn.Linear(embed_size, ff_hid_dim),
            nn.ReLU(),
            nn.Linear(ff_hid_dim, 4)  # Human bounding box (4 params: x, y, w, h)
        )
        
        self.ffn_bo = nn.Sequential(
            nn.Linear(embed_size, ff_hid_dim),
            nn.ReLU(),
            nn.Linear(ff_hid_dim, 4)  # Object bounding box (4 params: x, y, w, h)
        )
        
        self.ffn_phoi = nn.Sequential(
            nn.Linear(embed_size, ff_hid_dim),
            nn.ReLU(),
            nn.Linear(ff_hid_dim, num_act_classes)  # Interaction prediction vector
        )
        
        self.ffn_pobj = nn.Sequential(
            nn.Linear(embed_size, ff_hid_dim),
            nn.ReLU(),
            nn.Linear(ff_hid_dim, num_obj_classes)  # Object class prediction vector
        )
        
        # Output activation functions
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, decoder_ouput):
        
        B,Nq,F = decoder_ouput.shape #(Batch, no.queries, features)
        


        # Predictions (Feedforward networks)
        bh = self.ffn_bh(decoder_ouput)  # Human bounding box prediction (B, Nq, 4)
        bo = self.ffn_bo(decoder_ouput)  # Object bounding box prediction (B, Nq, 4)
        PHOI = self.ffn_phoi(decoder_ouput)  # Interaction prediction (B, Nq, Nact)
        Pobj = self.ffn_pobj(decoder_ouput)  # Object class prediction (B, Nq, Nobj)
        
        # Apply sigmoid to human, object bounding boxes and interaction prediction
        bh = self.sigmoid(bh)
        bo = self.sigmoid(bo)
        PHOI = self.sigmoid(PHOI)
        
        # Apply softmax to object class prediction
        Pobj = self.softmax(Pobj)
        
        # Interaction prediction weighting based on object class confidence
        max_obj_confidence = Pobj.max(dim=-1, keepdim=True)[0]  # (B, Nq, 1)
        PHOI = PHOI * max_obj_confidence  # Weighted interaction predictions
        
        out = {'pred_obj_logits': Pobj, 'pred_verb_logits': PHOI,
               'pred_sub_boxes': bh, 'pred_obj_boxes':bo}

        # return bh, bo, Pobj,  PHOI
        return out


