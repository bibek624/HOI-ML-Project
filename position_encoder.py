import torch.nn as nn
import torch


class PositionEncoder(nn.Module):
    """
    Positional Encoding for ResNet output in HOI.
    """
    def __init__(self, height=7, width=7, num_pos_feats=256):
        """
        Args:
            height (int): Height of the feature map (number of rows).
            width (int): Width of the feature map (number of columns).
            num_pos_feats (int): Number of positional afeatures (should be <= channel size) with rows and columns encoding combined.
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats

        # Learnable embeddings for row and column positions
        self.row_embed = nn.Embedding(height, num_pos_feats//2)
        self.col_embed = nn.Embedding(width, num_pos_feats//2)

        # Initialize weights for embeddings
        nn.init.uniform_(self.row_embed.weight, -0.1, 0.1)
        nn.init.uniform_(self.col_embed.weight, -0.1, 0.1)

    def forward(self, feature_map):
        """
        Args:
            feature_map (Tensor): ResNet feature map, shape (batch_size, channels, height, width).

        Returns:
            Tensor: Feature map with positional encodings added, shape (batch_size, channels, height, width).
        """
        batch_size, channels, height, width = feature_map.shape

        # Positional indices
        row_indices = torch.arange(height, device=feature_map.device)  # Shape (height,)
        col_indices = torch.arange(width, device=feature_map.device)   # Shape (width,)

        # Generate embeddings
        row_embedding = self.row_embed(row_indices)  # Shape (height, num_pos_feats)
        col_embedding = self.col_embed(col_indices)  # Shape (width, num_pos_feats)

        # Expand to form a grid
        row_embedding = row_embedding.unsqueeze(1).expand(-1, width, -1)  # Shape (height, width, num_pos_feats)
        col_embedding = col_embedding.unsqueeze(0).expand(height, -1, -1) # Shape (height, width, num_pos_feats)

        # Combine row and column embeddings
        pos_embedding = torch.cat([row_embedding, col_embedding], dim=-1) # Shape (height, width, 2*num_pos_feats)
        pos_embedding = pos_embedding.permute(2, 0, 1).unsqueeze(0)       # Shape (1, 2*num_pos_feats, height, width)

        # Expand positional encoding to match batch size
        pos_embedding = pos_embedding.expand(batch_size, -1, -1, -1)      # Shape (batch_size, 2*num_pos_feats, height, width)

        # # Add positional encoding to feature map
        # if channels > self.num_pos_feats * 2:
        #     # Pad positional embedding to match channels
        #     padding = torch.zeros(batch_size, channels - self.num_pos_feats * 2, height, width, device=feature_map.device)
        #     pos_embedding = torch.cat([pos_embedding, padding], dim=1)

        return feature_map + pos_embedding
