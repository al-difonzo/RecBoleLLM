# -*- coding: utf-8 -*-
# @Time   : 2020/10/6
# @Author : Yingqian Min
# @Email  : eliver_min@foxmail.com

r"""
ConvNCF
################################################
Reference:
    Xiangnan He et al. "Outer Product-based Neural Collaborative Filtering." in IJCAI 2018.

Reference code:
    https://github.com/duxy-me/ConvNCF
"""

import torch
import torch.nn as nn
import copy

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.layers import MLPLayers, CNNLayers
from recbole.model.general_recommender.bpr import BPR
from recbole.utils import InputType


class ConvNCFBPRLoss(nn.Module):
    """ConvNCFBPRLoss, based on Bayesian Personalized Ranking,

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = ConvNCFBPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self):
        super(ConvNCFBPRLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        distance = pos_score - neg_score
        loss = torch.sum(torch.log((1 + torch.exp(-distance))))
        return loss


class ConvNCF(GeneralRecommender):
    r"""ConvNCF is a a new neural network framework for collaborative filtering based on NCF.
    It uses an outer product operation above the embedding layer,
    which results in a semantic-rich interaction map that encodes pairwise correlations between embedding dimensions.
    We carefully design the data interface and use sparse tensor to train and test efficiently.
    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(ConvNCF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config["LABEL_FIELD"]

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.cnn_channels = config["cnn_channels"]
        self.cnn_kernels = config["cnn_kernels"]
        self.cnn_strides = config["cnn_strides"]
        self.dropout_prob = config["dropout_prob"]
        self.regs = config["reg_weights"]
        self.train_method = config["train_method"]
        self.pre_model_path = config["pre_model_path"]
        self.split_to = config.get("split_to", 0)

        # split the too large dataset into the specified pieces
        if self.split_to > 0:
            self.logger.info("split the n_items to {} pieces".format(self.split_to))
            self.group = torch.chunk(
                torch.arange(self.n_items).to(self.device), self.split_to
            )
        else:
            self.logger.warning(
                "Pay Attetion!! the `split_to` is set to 0. If you catch a OMM error in this case, "
                + "you need to increase it \n\t\t\tuntil the error disappears. For example, "
                + "you can append it in the command line such as `--split_to=5`"
            )

        # define layers and loss
        assert self.train_method in ["after_pretrain", "no_pretrain"]
        if self.train_method == "after_pretrain":
            assert self.pre_model_path != ""
            pretrain_state = torch.load(self.pre_model_path)["state_dict"]
            bpr = BPR(config=config, dataset=dataset)
            bpr.load_state_dict(pretrain_state)
            self.user_embedding = copy.deepcopy(bpr.user_embedding)
            self.item_embedding = copy.deepcopy(bpr.item_embedding)
        else:
            self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
            self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        self.cnn_layers = CNNLayers(
            self.cnn_channels, self.cnn_kernels, self.cnn_strides, activation="relu"
        )
        self.predict_layers = MLPLayers(
            [self.cnn_channels[-1], 1], self.dropout_prob, activation="none"
        )
        self.loss = ConvNCFBPRLoss()

    def forward(self, user, item):
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)

        interaction_map = torch.bmm(user_e.unsqueeze(2), item_e.unsqueeze(1))
        interaction_map = interaction_map.unsqueeze(1)

        cnn_output = self.cnn_layers(interaction_map)
        cnn_output = cnn_output.sum(axis=(2, 3))

        prediction = self.predict_layers(cnn_output)
        prediction = prediction.squeeze(-1)

        return prediction

    def reg_loss(self):
        r"""Calculate the L2 normalization loss of model parameters.
        Including embedding matrices and weight matrices of model.

        Returns:
            loss(torch.FloatTensor): The L2 Loss tensor. shape of [1,]
        """
        reg_1, reg_2 = self.regs[:2]
        loss_1 = reg_1 * self.user_embedding.weight.norm(2)
        loss_2 = reg_1 * self.item_embedding.weight.norm(2)
        loss_3 = 0
        for name, parm in self.cnn_layers.named_parameters():
            if name.endswith("weight"):
                loss_3 = loss_3 + reg_2 * parm.norm(2)
        for name, parm in self.predict_layers.named_parameters():
            if name.endswith("weight"):
                loss_3 = loss_3 + reg_2 * parm.norm(2)
        return loss_1 + loss_2 + loss_3

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        pos_item_score = self.forward(user, pos_item)
        neg_item_score = self.forward(user, neg_item)

        loss = self.loss(pos_item_score, neg_item_score)
        opt_loss = loss + self.reg_loss()

        return opt_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

    def full_sort_predict(self, interaction):
        r"""Full sort predict function. Given dataloader, return the scores of all items for all users.

        Args:
            interaction (DataLoader): a pytorch dataloader which contains (user_tensor, item_tensor) tuple.

        Returns:
            torch.FloatTensor: the scores of all items for all users. The shape is [n_users, n_items].
        """
        user_scores = []
        item_batches = []
        batch_size = interaction.batch_size

        # Iterate through dataloader to batch items
        with torch.no_grad():
            for batch_idx, (user_tensor, item_tensor) in enumerate(interaction):
                item_batches.append(item_tensor)

                # Predict scores if items are enough for a batch
                if len(item_batches) == batch_size or batch_idx == len(interaction) - 1:
                    item_tensor = torch.cat(item_batches, dim=0)
                    user_tensor = user_tensor.expand(item_tensor.size(0))

                    scores = self.forward(user_tensor, item_tensor)
                    user_scores.append(scores)

                    item_batches = []

        # Concatenate scores and reshape to match the output shape [n_users, n_items]
        user_scores = torch.cat(user_scores, dim=0)
        return user_scores.reshape(-1, self.n_items)


    # def full_sort_predict(self, interaction):
    #     user = interaction[self.USER_ID]
    #     item = interaction[self.ITEM_ID]
    #     user_e = self.user_embedding(user)
    #     item_e = self.item_embedding(item)

    #     interaction_map = torch.bmm(user_e.unsqueeze(2), item_e.unsqueeze(1))
    #     interaction_map = interaction_map.unsqueeze(1)

    #     cnn_output = self.cnn_layers(interaction_map)
    #     cnn_output = cnn_output.sum(axis=(2, 3))

    #     prediction = self.predict_layers(cnn_output)
    #     prediction = prediction.squeeze(-1)

    #     return prediction
    #     # user_inters = self.history_item_matrix[user]
    #     # item_nums = self.history_lens[user]
    #     scores = []

    #     # test users one by one, if the number of items is too large, we will split it to some pieces
    #     for user_input, item_num in zip(user_inters, item_nums.unsqueeze(1)):
    #         if self.split_to <= 0:
    #             output = self.user_forward(
    #                 user_input[:item_num], item_num, repeats=self.n_items
    #             )
    #         else:
    #             output = []
    #             for mask in self.group:
    #                 tmp_output = self.user_forward(
    #                     user_input[:item_num],
    #                     item_num,
    #                     repeats=len(mask),
    #                     pred_slc=mask,
    #                 )
    #                 output.append(tmp_output)
    #             output = torch.cat(output, dim=0)
    #         scores.append(output)
    #     result = torch.cat(scores, dim=0)
    #     return result
