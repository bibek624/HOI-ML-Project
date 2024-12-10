import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

class OAFeatureGenerator(nn.Module):
    def __init__(self, encoder_output_dim, num_oa_pairs, top_k):
        super(OAFeatureGenerator, self).__init__()
        self.num_oa_pairs = num_oa_pairs
        self.top_k = top_k

        # FFN for object-action candidate prediction
        self.g_cls = nn.Sequential(
            nn.Linear(encoder_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_oa_pairs),
            nn.Sigmoid()
        )

        # Pre-trained CLIP components for semantic embedding
        self.clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        # Linear projection to match image feature space
        self.semantic_projection = nn.Linear(self.clip_text_encoder.config.hidden_size, encoder_output_dim)

    def forward(self, encoder_features):
        # Step 1: Average pool encoder features
        avg_features = torch.mean(encoder_features, dim=1)  # Assuming input shape [B, T, D]

        # Step 2: Predict OA candidates
        oa_scores = self.g_cls(avg_features)  # Shape: [B, num_oa_pairs]

        # Step 3: Select top-K OA candidates
        top_k_values, top_k_indices = torch.topk(oa_scores, self.top_k, dim=1)

        # Step 4: Generate semantic embeddings for selected OA candidates
        sentences = self.generate_oa_sentences(top_k_indices)  # List of BxK sentences
        tokenized = self.clip_tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(encoder_features.device)
        clip_features = self.clip_text_encoder(**tokenized).last_hidden_state.mean(dim=1)  # Shape: [B*K, CLIP_dim]

        # Step 5: Project semantic embeddings to image feature space
        semantic_features = self.semantic_projection(clip_features)  # Shape: [B*K, encoder_output_dim]
        semantic_features = semantic_features.view(-1, self.top_k, encoder_features.size(-1))  # Reshape to [B, K, D]

        return semantic_features, top_k_indices

    def generate_oa_sentences(self, top_k_indices):
        # Transform OA pairs into sentences
        # Example: (phone, talk) -> "A person is talking on the phone"
        oa_pair_templates = [
            ("phone", "talk"), ("book", "read"), ("cup", "drink")  # Extend this list for all OA pairs
        ]
        sentences = []
        top_k_indices = torch.tensor([[0]])
        for batch_indices in top_k_indices:
            batch_sentences = []
            for idx in batch_indices:
                obj, action = oa_pair_templates[idx.item()]
                sentence = f"A person is {action} with a {obj}."
                batch_sentences.append(sentence)
            sentences.extend(batch_sentences)
        return sentences

# Example usage
batch_size = 8
time_steps = 10
encoder_output_dim = 512
num_oa_pairs = 50
top_k = 1

# Simulate encoder features
encoder_features = torch.randn(batch_size, time_steps, encoder_output_dim)

# Initialize and run the OA Feature Generator
oa_feature_generator = OAFeatureGenerator(encoder_output_dim, num_oa_pairs, top_k)
semantic_features, selected_oa_indices = oa_feature_generator(encoder_features)

print("Semantic Features Shape:", semantic_features.shape)  # [B, K, D]
print("Selected OA Indices:", selected_oa_indices)
