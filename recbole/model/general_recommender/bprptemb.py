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
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from recbole.utils import ModelType

# class BPRptemb(ContextRecommender):
class BPRptemb(GeneralRecommender):
    r"""BPRptemb is an extension of BPR that allows for pretrained item (user) embeddings."""
    input_type = InputType.PAIRWISE
    # model_type = ModelType.CONTEXT
    model_type = ModelType.GENERAL
    
    def __init__(self, config, dataset):
        super(BPRptemb, self).__init__(config, dataset)

        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.embedding_size = config['embedding_size']
        self.fine_tune = config['fine_tune']
        self.use_pretrained = config['use_pretrained']
        self.l2_regularization = config['l2_regularization']
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)

        if self.use_pretrained:
            pretrained_item_emb = dataset.get_preload_weight('iid')
            self.item_embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_item_emb).float(), freeze=not self.fine_tune)
        else:
            self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        
        self.loss = BPRLoss()

        self.apply(xavier_normal_initialization)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def calculate_loss(self, interaction):
      user = interaction[self.USER_ID]
      pos_item = interaction[self.ITEM_ID]
      neg_item = interaction[self.NEG_ITEM_ID]

      user_e = self.user_embedding(user)                        # [batch_size, embedding_size]
      pos_item_e = self.item_embedding(pos_item)                # [batch_size, embedding_size]
      neg_item_e = self.item_embedding(neg_item)                # [batch_size, embedding_size]
      pos_item_score = torch.mul(user_e, pos_item_e).sum(dim=1) # [batch_size]
      neg_item_score = torch.mul(user_e, neg_item_e).sum(dim=1) # [batch_size]

      loss = self.loss(pos_item_score, neg_item_score)          # []

      # Add regularization
      l2_norm = torch.tensor(0., device=self.device)
      for param in self.parameters():
          l2_norm += torch.norm(param)**2

      loss += self.l2_regularization * l2_norm

      return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_e = self.user_embedding(user)            # [batch_size, embedding_size]
        item_e = self.item_embedding(item)            # [batch_size, embedding_size]

        scores = torch.mul(user_e, item_e).sum(dim=1) # [batch_size]

        return scores
    
    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)                        # [batch_size, embedding_size]
        all_item_e = self.item_embedding.weight                   # [n_items, embedding_size]
        scores = torch.matmul(user_e, all_item_e.transpose(0, 1)) # [batch_size, n_items]
        return scores
