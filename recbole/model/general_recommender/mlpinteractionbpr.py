# -*- coding: utf-8 -*-
# @Time   :
# @Author :
# @Email  :

r"""
BPRptemb
################################################
"""

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
# from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.general_recommender import BPRptemb
# from recbole.model.init import xavier_normal_initialization
# from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from recbole.utils import ModelType

class MLPInteractionBPR(GeneralRecommender):
    input_type = InputType.PAIRWISE
    # model_type = ModelType.CONTEXT
    model_type = ModelType.GENERAL
    def __init__(self, config, dataset, fine_tune=False, use_pretrained=True, l2_regularization=0.01, hidden_layers=None):
        super(MLPInteractionBPR, self).__init__(config, dataset)
        print('Ciao mondo!')
        self = BPRptemb(config, dataset, fine_tune, use_pretrained, l2_regularization)
        print('TYPE OF SELF:', type(self))
        if hidden_layers is None:
            hidden_layers = [64]

        layers = []
        input_size = self.embedding_size * 2
        for output_size in hidden_layers:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
            input_size = output_size

        layers.append(nn.Linear(input_size, 1))

        self.interaction_net = nn.Sequential(*layers)

    def score(self, user_e, item_e):
        x = torch.cat([user_e, item_e], dim=1)
        scores = self.interaction_net(x).squeeze(1)
        return scores

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        scores = self.score(user_e, item_e)
        return scores

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        user_e = self.user_embedding(user)
        pos_item_e = self.item_embedding(pos_item)
        neg_item_e = self.item_embedding(neg_item)
        pos_item_score = self.score(user_e, pos_item_e)
        neg_item_score = self.score(user_e, neg_item_e)
        loss = self.loss(pos_item_score, neg_item_score)

        # Add regularization
        l2_norm = torch.tensor(0., device=self.device)
        for param in self.parameters():
            l2_norm += torch.norm(param)**2

        loss += self.l2_regularization * l2_norm

        return loss
    
    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)                        # [batch_size, embedding_size]
        all_item_e = self.item_embedding.weight                   # [n_items, embedding_size]
        scores = torch.matmul(user_e, all_item_e.transpose(0, 1)) # [batch_size, n_items]
        return scores
