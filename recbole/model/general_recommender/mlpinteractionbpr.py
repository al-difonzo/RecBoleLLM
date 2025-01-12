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
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from recbole.utils import ModelType

class MLPInteractionBPR(GeneralRecommender):
    r"""MLPInteractionBPR is a basic MLP model that uses BPR pairwise loss, and allows for pretrained item (user) embeddings."""
    input_type = InputType.PAIRWISE
    # model_type = ModelType.CONTEXT
    model_type = ModelType.GENERAL
    def __init__(self, config, dataset):
        super(MLPInteractionBPR, self).__init__(config, dataset)
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.embedding_size = config['embedding_size']
        self.fine_tune = config['fine_tune']
        self.use_pretrained = config['use_pretrained']
        self.l2_regularization = config['l2_regularization']
        self.hidden_layers = config.get('hidden_layers', [64])

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        if self.use_pretrained:
            pretrained_item_emb = dataset.get_preload_weight('iid')
            self.item_embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_item_emb).float(), freeze=not self.fine_tune)
        else:
            self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()
        # self.hidden_layers = [64] if self.hidden_layers is None else self.hidden_layers
        self.apply(xavier_normal_initialization)

        input_size = self.embedding_size * 2
        self.layers = []
        for output_size in self.hidden_layers:
            self.layers.append(nn.Linear(input_size, output_size))
            self.layers.append(nn.ReLU())
            input_size = output_size

        self.layers.append(nn.Linear(input_size, 1))
        self.interaction_net = nn.Sequential(*self.layers)

    def score(self, user_e, item_e):
        x = torch.cat([user_e, item_e], dim=1)
        scores = self.interaction_net(x).squeeze(1)
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

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        scores = self.score(user_e, item_e)
        return scores
    
    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)                        # [batch_size, embedding_size]
        all_item_e = self.item_embedding.weight                   # [n_items, embedding_size]
        scores = torch.matmul(user_e, all_item_e.transpose(0, 1)) # [batch_size, n_items]
        return scores
