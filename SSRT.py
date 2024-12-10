

import torch
import torch.nn as nn
import Backbone
import position_encoder
from Encoder import Encoder
import OA_Candidate_Sampler
import SpatialFeatureGenerator 
import Semantic_Feature_Generator 
import Feature_Aggregator
import Query_Refiner
import HOI_Decoder
import HOI_Prediction
from importlib import reload

reload(SpatialFeatureGenerator)
reload(Semantic_Feature_Generator)
reload(Backbone)
reload(HOI_Prediction)


class SSRT(nn.Module):
    def __init__(self, num_oas, batch_size, nobj, nact, device, OA_distn, n_queries):
        super(SSRT, self).__init__()
        
        self.device = device
        # Step 1: Initialize modules
        self.backbone = Backbone.Backbone(device=self.device).to(self.device)
        self.position_encoder = position_encoder.PositionEncoder().to(self.device)
        self.encoder = Encoder().to(self.device)
        self.oas_sampler = OA_Candidate_Sampler.OACandidateSampler(num_oas).to(self.device)
        self.spatial_feature_generator = SpatialFeatureGenerator.SpatialFeatureGenerator(OA_distn=OA_distn,device=self.device).to(self.device)
        self.semantic_feature_generator = Semantic_Feature_Generator.SemanticFeatureGenerator(OA_distn=OA_distn,device=self.device).to(self.device)
        self.feature_aggregator = Feature_Aggregator.FeatureAggregator().to(self.device)
        self.query_refiner = Query_Refiner.QueryRefiner(batch_size, num_queries=n_queries).to(self.device)
        self.hoi_decoder = HOI_Decoder.HOIDecoder().to(self.device)
        self.hoi_predictor = HOI_Prediction.HOIPrediction(num_act_classes=nact, num_obj_classes=nobj).to(self.device)

    def forward(self, imgs):
        # Step 1: Backbone
        features = self.backbone(imgs)
        
        features, pos = self.backbone()
        # Step 2: Position Encoder
        position_encoded = self.position_encoder(features)
        
        # Step 3: Vectorize Feature Map
        vectorized_features = self.grid2vector(position_encoded)
        
        # Step 4: Encoder
        encoded_features = self.encoder(vectorized_features)
        
        # Step 5: OA Candidate Sampler
        oas = self.oas_sampler(encoded_features)
        
        # Step 6: Spatial Feature Generator
        spatial_features = self.spatial_feature_generator(oas)
        
        # Step 7: Semantic Feature Generator
        semantic_features = self.semantic_feature_generator(oas)
        
        # Step 8: Feature Aggregator
        aggregated_features = self.feature_aggregator(spatial_features, semantic_features)
        
        # Step 9: Query Refiner
        refined_query = self.query_refiner(aggregated_features)
        
        # Step 10: HOI Decoder
        decoded_query = self.hoi_decoder(encoded_features, refined_query)
        
        # Step 11: HOI Prediction
        predictions = self.hoi_predictor(decoded_query)
        
        return predictions

    def grid2vector(self,feature_map):
        B,C,W,H = feature_map.shape # (Batch, Channels, Width, Height)
        feature_map = feature_map.reshape((B,C,W*H)) #changing 2d image to vector (B, C, F) (F = feature)
        feature_map = feature_map.permute([0,2,1]) # (B, F, C)
        
        return feature_map
