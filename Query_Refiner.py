import torch
import torch.nn as nn

class QueryRefiner(nn.Module):
    def __init__(self,batch_size, query_dim = 256, support_feature_dim = 256, num_queries = 49):
        super(QueryRefiner, self).__init__()
        self.query_dim = query_dim
        self.support_feature_dim = support_feature_dim
        self.num_queries = num_queries

        self.queries = nn.Parameter(torch.randn(batch_size,num_queries, query_dim))

        self.self_attention = nn.MultiheadAttention(embed_dim=query_dim, num_heads=8, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim=query_dim, num_heads=8, batch_first=True)

        self.layer_norm = nn.LayerNorm(query_dim)

    def forward(self, support_features):
        """
        Args:
            support_features (torch.Tensor): Tensor of support features, shape (seq_len, batch_size, support_feature_dim)
        
        Returns:
            torch.Tensor: Refined queries, shape (num_queries, batch_size, query_dim)
        """
        # Self-attention
        q_self_attn, _ = self.self_attention(self.queries, self.queries, self.queries)
        q_self_attn = self.layer_norm(self.queries + q_self_attn)

        # Cross-attention
        q_cross_attn, _ = self.cross_attention(q_self_attn, support_features, support_features)

        return q_cross_attn