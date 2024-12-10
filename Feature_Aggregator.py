import torch
import torch.nn as nn

class FeatureAggregator(nn.Module):
    def __init__(self, feature_dim = 256, *args, **kwargs):
        """
        Initializes the feature aggregator.
        :param semantic_dim: Dimension of semantic features.
        :param spatial_dim: Dimension of spatial features.
        :param aggregated_dim: Dimension of aggregated features after processing.
        """
        super().__init__(*args, **kwargs)

        # Concatenate semantic and spatial features, then project to aggregated_dim
        self.aggregator = nn.Linear(feature_dim*2 , feature_dim)

    def forward(self, semantic_features, spatial_features):
        """
        Aggregates semantic and spatial features for each candidate.
        :param semantic_features: Tensor of shape (B, K, semantic_dim), semantic features for all candidates.
        :param spatial_features: Tensor of shape (B, K, spatial_dim), spatial features for all candidates.
        :return: Tensor of shape (B, K, aggregated_dim), aggregated features for all candidates.
        """
        # Concatenate along the feature dimension
        concatenated_features = torch.cat([semantic_features, spatial_features], dim=-1)

        # Aggregate using the linear layer
        aggregated_features = self.aggregator(concatenated_features)

        return aggregated_features
