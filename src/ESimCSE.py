import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig, BertTokenizer


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2",
                                    "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 *
                             attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 *
                             attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


# code reference: https://github.com/shuxinyin/SimCSE-Pytorch
class ESimCSEModel(nn.Module):

    def __init__(self, pretrained_model, pooler_type, dropout=0.3):
        super(ESimCSEModel, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = dropout 
        config.hidden_dropout_prob = dropout
        config.pooler_type = pooler_type
        self.encoder = BertModel.from_pretrained(pretrained_model, config=config)
        #self.pooling = pooling
        self.pooler = Pooler(config.pooler_type)
        if config.pooler_type == "cls":
            self.mlp = MLPLayer(config)


    def save_model(self, path):
        # save bert model, bert config
        self.encoder.save_pretrained(path)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.encoder(input_ids, attention_mask,
                        token_type_ids, output_hidden_states=True)

        pooler_output = self.pooler(attention_mask, outputs)

        # If using "cls", we add an extra MLP layer
        # (same as BERT's original implementation) over the representation.
        if self.pooler.pooler_type == "cls":
            pooler_output = self.mlp(pooler_output)

        return pooler_output

        # if self.pooling == 'cls':
        #     return out.last_hidden_state[:, 0]  # [batch, 768]
        # if self.pooling == 'pooler':
        #     return out.pooler_output  # [batch, 768]
        # if self.pooling == 'last-avg':
        #     last = out.last_hidden_state.transpose(
        #         1, 2)  # [batch, 768, seqlen]
        #     # [batch, 768]
        #     return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
        # if self.pooling == 'first-last-avg':
        #     first = out.hidden_states[1].transpose(
        #         1, 2)  # [batch, 768, seqlen]
        #     # [batch, 768, seqlen]
        #     last = out.hidden_states[-1].transpose(1, 2)
        #     first_avg = torch.avg_pool1d(
        #         first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        #     last_avg = torch.avg_pool1d(
        #         last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        #     # [batch, 2, 768]
        #     avg = torch.cat((first_avg.unsqueeze(
        #         1), last_avg.unsqueeze(1)), dim=1)
        #     # [batch, 768]
        #     return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)


class MomentumEncoder(ESimCSEModel):
    """ MomentumEncoder """

    def __init__(self, pretrained_model, pooling, dropout):
        super(MomentumEncoder, self).__init__(
            pretrained_model, pooling, dropout)


class MultiNegativeRankingLoss(nn.Module):
    # code reference: https://github.com/zhoujx4/NLP-Series-sentence-embeddings
    def __init__(self):
        super(MultiNegativeRankingLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def multi_negative_ranking_loss(self, embed_src, embed_pos, embed_neg, scale=20.0):
        '''
        scale is a temperature parameter
        '''

        if embed_neg is not None:
            embed_pos = torch.cat([embed_pos, embed_neg], dim=0)

        # print(embed_src.shape, embed_pos.shape)
        scores = self.cos_sim(embed_src, embed_pos) * scale

        labels = torch.tensor(range(len(scores)),
                              dtype=torch.long,
                              device=scores.device)  # Example a[i] should match with b[i]

        return self.cross_entropy_loss(scores, labels)

    def cos_sim(self, a, b):
        """ the function is same with torch.nn.F.cosine_similarity but processed the problem of tensor dimension
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))


# if __name__ == '__main__':
#     import numpy as np

#     input1 = torch.randn(100, 128)
#     input2 = torch.randn(100, 128)
#     output = F.cosine_similarity(input1, input2)
#     print(output.shape)

#     embed_src = torch.tensor(np.random.randn(32, 768))  # (batch_size, 768)
#     embed_pos = torch.tensor(np.random.randn(32, 768))
#     embed_neg = torch.tensor(np.random.randn(160, 768))

#     ESimCSELoss = MultiNegativeRankingLoss()
#     esimcse_loss = ESimCSELoss.multi_negative_ranking_loss

#     res = esimcse_loss(embed_src, embed_pos, embed_neg)
