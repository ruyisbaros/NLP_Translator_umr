import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model  # embedding dimension
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # seq_len = how many tokens at a time each encoder layer processes, d_model = embedding dimension
        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a matrix of shape (seq_len, 1)
        position = torch.arange(0, seq_len).unsqueeze(1)  # (seq_len, 1)
        # calculate positional encoding formula for sin and cosine
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # apply sin for even indices and cosine for odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # reshape to (seq_len, d_model)
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # add positional encoding to the input tensor
        x = x + (self.pe[:, : x.shape[1], :]).require_grad(
            False
        )  # (batch_size, seq_len, d_model)
        return self.dropout(x)


class LayerNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # multiplier
        self.bias = nn.Parameter(torch.zeros(1))  # additive

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)  # W2 and B2

    def forward(self, x):
        # input x is (batch_size, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)  # WQ
        self.k_linear = nn.Linear(d_model, d_model)  # WK
        self.v_linear = nn.Linear(d_model, d_model)  # WV

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(q, k, v, mask, dropout=nn.Dropout):
        head_dim = q.shape[-1]
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = F.softmax(attention_scores, dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        x = torch.matmul(
            attention_scores, v
        )  # (batch_size, seq_len, num_heads, head_dim) => (batch_size, seq_len, d_model)

        return x, attention_scores

    def forward(self, query, key, value, mask):
        # linearly transform query, key, and value
        q = self.q_linear(query)  # (batch_size, seq_len, d_model)
        k = self.k_linear(key)  # (batch_size, seq_len, d_model)
        v = self.v_linear(value)  # (batch_size, seq_len, d_model)

        # (batch_size, seq_len, d_model) => (batch_size, seq_len, num_heads, head_dim) => (batch_size, num_heads, seq_len, head_dim)
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim).transpose(
            1, 2
        )
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).transpose(
            1, 2
        )

        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            q, k, v, mask, self.dropout
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], -1, self.num_heads * self.head_dim)
        )

        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout,
    ):
        super().__init__()
        self.attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x
