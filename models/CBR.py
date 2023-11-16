import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math


class CrossEn(nn.Module):
    def __init__(self):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss

class ContraAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg.model.transformer_encoder.att_dim
        self.num_attention_heads = cfg.model.transformer_encoder.att_head
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (self.hidden_size, self.num_attention_heads))
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.logit_scale = nn.Parameter(torch.ones([]))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, attentin_mask):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:

        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        # attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
        batch, L, D = key_states.size()
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)

        mask = torch.Tensor(np.ones([batch, L])).cuda()
        mask_sum = mask.sum(-1)
        com_mask_sum = attentin_mask.sum(-1)
        retrieve_logits = torch.einsum("ahld,bhmd->ablm", query_layer, key_layer)  # (B,B,Nq,Nk)
        t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # B,B,Nq,Nk -> B,B,Nq
        v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # B,B,Nq,Nk -> B,B,Nk
        t2v_logits = torch.sum(t2v_logits, dim=2) / (com_mask_sum.unsqueeze(1))
        v2t_logits = torch.sum(v2t_logits, dim=2) / (mask_sum.unsqueeze(0))
        retrieve_logits = (t2v_logits + v2t_logits) / 2.0
        logit_scale = self.logit_scale.exp()
        retrieve_logits = logit_scale * retrieve_logits

        return retrieve_logits, retrieve_logits.T

class CBR(nn.Module):

  def __init__(self, cfg):
    super().__init__()
    embed_dim = 512
    self.conv1 = nn.Conv2d(embed_dim * 2, embed_dim, 1)

    # the number of heads is tunable
    self.mh_att = nn.MultiheadAttention(embed_dim, num_heads=8, bias=False)
    self.conv2 = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 1),
                               )

    self.contra = ContraAttention(cfg)

    self.loss = CrossEn()

  def forward(self, bef_feat, sent_feat, aft_feat, masks, soft_triplet_loss=True):
      batch, max_v_len, d_feat = bef_feat.shape[:]
      max_l_len = masks.shape[-1]

      masks_new = masks.view(batch, max_l_len, 1).expand(-1, -1, d_feat)
      sent_feat = sent_feat * masks_new
      sent_feat_mean = sent_feat.mean(1)
      bef_feat = bef_feat.permute(0, 2, 1).view(batch, d_feat, 14, 14)
      text_feat = sent_feat_mean.view(batch, d_feat, 1, 1).expand(-1, -1, 14, 14)

      v1_feat = torch.cat([bef_feat, text_feat], 1)  # B x 2D x H x W
      v1_feat = self.conv1(v1_feat)  # B x D x H x W
      v1_feat = v1_feat.view(batch, d_feat, 14 * 14).permute(2, 0, 1)  # H*W x B x D
      self_att, _ = self.mh_att(v1_feat, v1_feat, v1_feat)  # H*W x B x D
      self_att = self_att.view(14, 14, batch, d_feat).permute(2, 3, 0,
                                                   1)  # B x D x H x W
      self_att = self.conv2(self_att)  # B x D x H x W

      vid_mask = torch.Tensor(np.ones([batch, max_v_len])).cuda()

      mod_img1 = self_att.view(batch, d_feat, -1).permute(0, 2, 1)

      contra_score1, contra_score2 = self.contra(mod_img1, aft_feat, vid_mask)
      loss = (self.loss(contra_score1) + self.loss(contra_score2)) / 2.0

      return loss

