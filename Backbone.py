import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image



class Backbone(nn.Module):

    #input should be pillow images stacked along batch dimension 
    
    def __init__(self,device, reshape_dim = 256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reshape_dim = 256
        
        self.device = device
        
        resnet =  models.resnet50()
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-2]) #removing the last avg pool and fc layer
        
        self.transform = transforms.Compose([
                transforms.Resize(256), # Resize to 256x256
                transforms.CenterCrop(224), # Crop to 224x224
                transforms.ToTensor(), # Convert to tensor
                transforms.Normalize(  # Normalize according to ImageNet statistics
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
                    ])
        
        self.reshape_conv = nn.Conv2d(2048,reshape_dim,1)
        
        for p in self.parameters():
                p.requires_grad_(False)
    
    @torch.no_grad()
    def forward(self, images):
        
        transformed_images = []
        for image in images:
            transformed_image = self.transform(image)
            transformed_images.append(transformed_image)
        
        transformed_images = torch.stack(transformed_images, dim=0).to(self.device) # (batch, 3, 224, 224)
        
        output = self.backbone(transformed_images) #(batch, 2048, 7, 7)
        reshaped_output = self.reshape_conv(output) # (batch, reshape_dim,7,7)

        return reshaped_output
     


