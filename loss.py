import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class MultiViewSimilarityLoss(nn.Module):
    def __init__(self):
        super(MultiViewSimilarityLoss, self).__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, embeddings, negative):
        similarity_loss = []
        for i in range(0, len(embeddings)):
            cur_embed = embeddings[i]
            for j in range(i + 1, len(embeddings)):
                # print(embeddings[j].shape, cur_embed.shape, negative.shape)
                # dist_a = F.pairwise_distance(dist_a, embedding[i], p=2)
                # dist_b = F.pairwise_distance(embedding[i-1], negative, p=2)
                # When nn.TripletMarginLoss, the pairwise distance metric is computed within the function
                sim_loss = self.triplet_loss(cur_embed, embeddings[j], negative)
                similarity_loss.append(sim_loss)
        if 0 != len(similarity_loss):
            sim_loss = torch.mean(torch.stack(similarity_loss))
            return sim_loss
        return self.triplet_loss(embeddings[0], embeddings[0], negative)
