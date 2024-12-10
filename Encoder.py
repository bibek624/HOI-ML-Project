import torch.nn as nn



class Encoder(nn.Module):
    
    def __init__(self, d_model = 256, nhead = 8, nlayers = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoderLayer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True,norm_first=True)
        self.encoder = nn.TransformerEncoder(self.encoderLayer, nlayers, enable_nested_tensor=False)
        
    def forward(self, x):
        
        encoded_ouput = self.encoder(x)
        
        return encoded_ouput