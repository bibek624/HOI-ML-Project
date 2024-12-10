import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

class SpatialFeatureGenerator(nn.Module):
    def __init__(self,
                 device,
                 OA_distn,
                 k = 1,
                 grid_size=100,
                 feature_dim=256,
                 spatial_embed_dim=64,

                 ):
        """
        Trainable Spatial Feature Generator

        Args:
            k = number of topk
            num_oa_labels (int): Number of Object-Action (OA) labels
            grid_size (int): Size of spatial map grid
            feature_dim (int): Dimension of output spatial features
            spatial_embed_dim (int): Intermediate embedding dimension
        """
        super().__init__()
        self.device = device
        self.OA_distn = OA_distn
        # self.spatial_means = 1
        # self.spatial_logvars = 0
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1,padding=1),
            # nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3,stride=1, padding=1),
            # nn.ReLU()
        )
        
        self.grid_size = grid_size
        # Linear projection layer
        self.projection = nn.Linear(32 * self.grid_size**2, feature_dim)
        

        self.grid_size = grid_size

    #     self.load_distribution()

        
    # def load_distribution(self):
    #     with open(self.distribution_file, 'r') as f:
    #         self.id2distn = json.load(f)
            
    #     # Convert 'mean_xy' and 'cov_xy' to tensors
    #     for key, value in self.id2distn.items():
    #         # Convert mean and covariance for the 'xy' distribution
    #         value['mean_xy'] = torch.tensor(value['mean_xy'], dtype=torch.float32).to(self.device)
    #         value['cov_xy'] = torch.tensor(value['cov_xy'], dtype=torch.float32).to(self.device)
            
    #         # Convert mean and covariance for the 'wh' distribution
    #         value['mean_wh'] = torch.tensor(value['mean_wh'], dtype=torch.float32).to(self.device)
    #         value['cov_wh'] = torch.tensor(value['cov_wh'], dtype=torch.float32).to(self.device)


    
    def forward(self,OA_ids):
        """
        Forward pass to generate spatial features

        Args:
            human_bbox (torch.Tensor): Human bounding box
            oa_label (torch.Tensor): Object-Action label

        Returns:
            torch.Tensor: Generated spatial features
        """
        
        binary_maps = self.generate_binary_map(OA_ids) #(B, K, 2, grid_size, grid_size)
        
        # Reshape to (B * K, 2, grid_size, grid_size) for convolution
        B, K, C, H, W = binary_maps.shape
        binary_maps = binary_maps.view(B * K, C, H, W)  # Shape: (B * K, 2, grid_size, grid_size)

      
        # Apply convolutional layers
        conv_features = self.conv_layers(binary_maps)  # Shape: (B * K, 32, grid_size, grid_size)

        # Optional: Reshape back to (B, K, ...)
        # conv_features = conv_features.view(B, K, conv_features.size(1), H, W)
        
        # Flatten for projection layer
        conv_features = conv_features.view(conv_features.size(0), -1)  # Shape: (B * K, 32 * grid_size * grid_size)

        # Apply projection layer
        projected_features = self.projection(conv_features)  # Shape: (B * K, feature_dim)

        # Reshape back to (B, K, feature_dim) for output
        projected_features = projected_features.view(B, K, -1)

        
    
        return projected_features



    def sample_from_distribution(self, id):

       data = self.OA_distn 
       mean_xy = data[id]['mean_xy']
       var_xy = data[id]['cov_xy']

       mean_wh = data[id]['mean_wh']
       var_wh = data[id]['cov_wh']

    
    #    var = torch.tensor([[0.5,0.2],[0.2,0.3]]).to(self.device)
    #    print(var_wh)

       distribution1 = torch.distributions.MultivariateNormal(mean_xy, var_xy)
       distribution2 = torch.distributions.MultivariateNormal(mean_wh, var_wh)

       dx,dy = distribution1.sample((1,)).reshape(2,)
       dw, dh = distribution2.sample((1,)).reshape(2,)

       return dx,dy,dw,dh
    
    def get_rsc_batch(self, ids):
        """
        Retrieve relative spatial coordinates for a batch of IDs.

        Args:
            ids (torch.Tensor): Tensor of IDs with shape (B, K)

        Returns:
            tuple: Tensors (delta_x, delta_y, delta_w, delta_h) with shape (B, K)
        """
        batch_size, num_ids = ids.shape
        
        # Ensure these tensors are on the correct device
        delta_x = torch.zeros(batch_size, num_ids).to(self.device)
        delta_y = torch.zeros(batch_size, num_ids).to(self.device)
        delta_w = torch.zeros(batch_size, num_ids).to(self.device)
        delta_h = torch.zeros(batch_size, num_ids).to(self.device)

        for b in range(batch_size):
            for k in range(num_ids):
                # Example values (you might have actual data for these)

                id = ids[b][k].item()
                dx, dy, dw, dh = self.sample_from_distribution(id)
                # dx, dy, dw, dh = 0.5, 0.5, 0.5, 0.5
                
                delta_x[b, k] = dx
                delta_y[b, k] = dy
                delta_w[b, k] = dw
                delta_h[b, k] = dh

        return delta_x, delta_y, delta_w, delta_h
        
    def generate_binary_map(self, ids):
        """
        Create binary matrices for human and object bounding boxes for a batch of IDs
        and scale them to 1000x1000.

        Args:
            ids (torch.Tensor): Tensor of IDs with shape (B, K)

        Returns:
            torch.Tensor: Tensor of scaled binary maps with shape (B, K, 2, 1000, 1000)
        """
        delta_x, delta_y, delta_w, delta_h = self.get_rsc_batch(ids)

        human_width = human_height = 100
        batch_size, num_ids = ids.shape

        # Compute bounding box parameters
        obj_x = (delta_x * human_width).int()
        obj_y = (delta_y * human_height).int()
        obj_width = (human_width * torch.exp(delta_w)).int()
        obj_height = (human_height * torch.exp(delta_h)).int()

        # Calculate global bounding box dimensions
        global_x1 = torch.min(torch.zeros_like(obj_x), obj_x)
        global_y1 = torch.min(torch.zeros_like(obj_y), obj_y)
        global_x2 = torch.max(human_width + torch.zeros_like(obj_x), obj_x + obj_width)
        global_y2 = torch.max(human_height + torch.zeros_like(obj_y), obj_y + obj_height)

        global_width = (global_x2 - global_x1).max().item()
        global_height = (global_y2 - global_y1).max().item()

        # Initialize binary maps
        human_binary_maps = torch.zeros((batch_size, num_ids, global_height, global_width)).to(self.device)
        object_binary_maps = torch.zeros((batch_size, num_ids, global_height, global_width)).to(self.device)

        # Populate binary maps
        for b in range(batch_size):
            for k in range(num_ids):
                human_binary_maps[b, k, :human_height, :human_width] = 1
                object_binary_maps[b, k, obj_y[b, k]:obj_y[b, k] + obj_height[b, k],
                                obj_x[b, k]:obj_x[b, k] + obj_width[b, k]] = 1

        # Combine maps into a single tensor
        binary_maps = torch.stack((human_binary_maps, object_binary_maps), dim=2)  # Shape: (B, K, 2, H, W)

        # Reshape to (N, C, H, W) for interpolation
        binary_maps = binary_maps.view(-1, 2, global_height, global_width)  # Flatten B and K into batch dimension

        # Scale to gridsize x gridsize
        scaled_binary_maps = F.interpolate(binary_maps, size=(self.grid_size, self.grid_size), mode='bilinear', align_corners=False)

        # Reshape back to (B, K, 2, gridsize, gridsize)
        scaled_binary_maps = scaled_binary_maps.view(batch_size, num_ids, 2, self.grid_size, self.grid_size)

        return scaled_binary_maps
        
    

    
    
    