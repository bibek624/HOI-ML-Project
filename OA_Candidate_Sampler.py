import torch
import torch.nn as nn


class OACandidateSampler(nn.Module):
    def __init__(self, num_oas, input_dim = 256, hidden_dim = 2048, topk = 5):
        """
        Args:
            input_dim: Dimension of the input encoded features Fe(x)
            hidden_dim: Hidden dimension for the feedforward network gcls
            num_oas: Number of possible object-action pairs (Ns)
            top_k: Number of top candidates to select
        """
        super(OACandidateSampler, self).__init__()
        
        # Define a 3-layer Feedforward Network (FFN) gcls
        self.gcls = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_oas)  # Output the OA candidate logits
        )
        
        self.topk = topk  # number of oa candidates to sample
       

    def forward(self, encoded_features):
        """
        Args:
            encoded_features: The input feature tensor Fe(x) with shape (batch_size, num_channels, height, width)
        
        Returns:
            Scand: Top-K OA candidates (yo, ya) for each batch
        """
        
        B,T,C = encoded_features.shape
        
        # Average pool the encoded features to (batch_size, num_channels)
        avg_pooled = torch.mean(encoded_features, dim=1)
        
        # Pass the pooled features through the gcls network to get OA pair scores
        oa_logits = self.gcls(avg_pooled)  # Shape: (batch_size, num_oas)
        
        # # Apply sigmoid to get probabilities
        oa_probabilities = torch.sigmoid(oa_logits)  # Shape: (batch_size, num_oas)

        probs, topKIndices = torch.topk(oa_probabilities,self.topk)

        
        return topKIndices

