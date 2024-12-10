import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
import torch



class SemanticFeatureGenerator(nn.Module):
    def __init__(self, device, OA_distn, output_dim=256, *args, **kwargs):
        super().__init__(*args, **kwargs)

         # Define device
        self.device = device
        self.OA_distn = OA_distn
        # Load CLIP components
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # Set up projection layer
        clip_output_dim = self.text_encoder.config.hidden_size  # Fetch the hidden size
        self.semantic_projection = nn.Linear(clip_output_dim, output_dim)

    def compute_semantic_feature(self, oa_candidate):
        """
        Compute semantic feature for a single OA candidate.
        :param oa_candidate: Tuple (action, object), e.g., ('hit', 'ball')
        :return: Projected semantic feature tensor.
        """
        # Convert OA candidate to descriptive sentence
        sentence = f"A person is {oa_candidate[1]} with a {oa_candidate[0]}."
        tokenized_input = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Compute sentence embedding using CLIP's text encoder
        sentence_embedding = self.text_encoder(**tokenized_input).last_hidden_state[:, 0, :]
        
        # Project to semantic feature space
        semantic_feature = self.semantic_projection(sentence_embedding)
        return semantic_feature

    def get_oa_candidate(self, id):

        OA_pair = self.OA_distn[id]['OA_pair']
        return OA_pair

    def forward(self, oa_ids):
        """
        Compute semantic features for a batch of OA candidates.
        :param oa_ids: Tensor of size (B, K), where B is batch size and K is the number of OA candidates per batch.
        :return: Tensor of size (B, K, output_dim), semantic features for all candidates.
        """
        B, K = oa_ids.shape
        output_dim = self.semantic_projection.out_features

        # Placeholder for semantic features
        semantic_features = torch.zeros(B, K, output_dim, device=self.device)

        # Map IDs to OA candidates (assumes id2OaCandidate is defined externally)
        for b in range(B):
            for k in range(K):
                oa_id = oa_ids[b, k].item()  # Fetch the ID
                # oa_candidate = id2OaCansdidate[oa_id]  # Map ID to OA candidate (action, object)
                oa_candidate = self.get_oa_candidate(oa_id)
                semantic_features[b, k] = self.compute_semantic_feature(oa_candidate)

        return semantic_features
