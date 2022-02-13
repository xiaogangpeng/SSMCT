# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.
# 自注意力，多头自注意力机制，前馈神经网络层

""" Define the attention layers. """
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2) # dim = -1，对最后一维做softmax

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))  # 计算两个tensor的矩阵乘法，注意维度必须为3,注意k的维度
        attn = attn / self.temperature
        if mask is not None:
            batch_size, _, _ = attn.size()
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)   # F.softmax
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias = False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias = False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias = False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))   # 按照正态分布初始化
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model, bias = False)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        '''
        q,k,v: [batch_size, len_q, d_model]，
        传3次一样的参数进来
        '''

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q    # 残差连接，记录初始输入

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k) # [batch_size, len_q, n_head, d_k]
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k) # [batch_size, len_k, n_head, d_k]
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v) # [batch_size, len_v, n_head, d_v]

        # 先调换维度，再转为连续数据，最后用view
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        # mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)         #output is final output, attn is score

        output = output.view(n_head, sz_b, len_q, d_v)  # n_head, sz_b, len_q, d_v
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))  # b * lq * d_model
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):   # d_model, d_inner
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid, bias=False)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in, bias=False)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
