# model_utils.py
# Standard PyTorch architectures you can reuse across projects/interviews.
#
# Conventions:
# - For classification models, outputs are LOGITS (not probabilities).
#   Use BCEWithLogitsLoss / CrossEntropyLoss and apply sigmoid/softmax only for reporting.
# - For regression, outputs are raw predictions (shape (B, d_y)).
#
# All modules are torch.nn.Module and are intentionally minimal + readable.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union

import torch
from torch import nn


TaskType = Literal["regression", "binary", "multiclass"]


def init_linear(layer: nn.Linear, gain: float = 1.0) -> None:
  nn.init.xavier_uniform_(layer.weight, gain=gain)
  if layer.bias is not None:
    nn.init.zeros_(layer.bias)


# ----------------------------
# 1) Linear / Logistic / Softmax regression
# ----------------------------

class LinearRegression(nn.Module):
  """
  Linear regression (possibly multi-output).
  Input:  x: (B, d_in)
  Output: y_hat: (B, d_out)
  """
  def __init__(self, d_in: int, d_out: int = 1, bias: bool = True):
    super().__init__()
    self.linear = nn.Linear(d_in, d_out, bias=bias)
    init_linear(self.linear)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear(x)


class LogisticRegression(nn.Module):
  """
  Binary logistic regression (outputs logits).
  Input:  x: (B, d_in)
  Output: logits: (B, 1)
  """
  def __init__(self, d_in: int, bias: bool = True):
    super().__init__()
    self.linear = nn.Linear(d_in, 1, bias=bias)
    init_linear(self.linear)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear(x)


class SoftmaxRegression(nn.Module):
  """
  Multinomial logistic regression (softmax regression; outputs logits).
  Input:  x: (B, d_in)
  Output: logits: (B, n_classes)
  """
  def __init__(self, d_in: int, n_classes: int, bias: bool = True):
    super().__init__()
    if n_classes < 2:
      raise ValueError("n_classes must be >= 2")
    self.linear = nn.Linear(d_in, n_classes, bias=bias)
    init_linear(self.linear)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear(x)


# ----------------------------
# 2) MLP for regression/binary/multiclass
# ----------------------------

class MLP(nn.Module):
  """
  Generic MLP head for:
  - regression:  outputs (B, d_out)
  - binary:      outputs (B, 1) logits
  - multiclass:  outputs (B, n_classes) logits

  Input: x: (B, d_in)
  """
  def __init__(
    self,
    d_in: int,
    task: TaskType,
    d_out: int = 1,                 # for regression: number of target dims
    n_classes: Optional[int] = None, # for multiclass
    hidden_sizes: Sequence[int] = (256, 128),
    dropout: float = 0.1,
    activation: Literal["relu", "gelu", "tanh"] = "relu",
    layer_norm: bool = False
  ):
    super().__init__()
    self.task = task

    if task == "multiclass":
      if n_classes is None or n_classes < 2:
        raise ValueError("For multiclass, provide n_classes >= 2")
      out_dim = int(n_classes)
    elif task == "binary":
      out_dim = 1
    elif task == "regression":
      if d_out < 1:
        raise ValueError("For regression, d_out must be >= 1")
      out_dim = int(d_out)
    else:
      raise ValueError(f"Unknown task: {task}")

    if activation == "relu":
      act = nn.ReLU()
      gain = nn.init.calculate_gain("relu")
    elif activation == "gelu":
      act = nn.GELU()
      gain = 1.0
    elif activation == "tanh":
      act = nn.Tanh()
      gain = nn.init.calculate_gain("tanh")
    else:
      raise ValueError("activation must be one of: relu, gelu, tanh")

    layers = []
    prev = d_in
    for h in hidden_sizes:
      lin = nn.Linear(prev, int(h))
      init_linear(lin, gain=gain)
      layers.append(lin)
      if layer_norm:
        layers.append(nn.LayerNorm(int(h)))
      layers.append(act)
      if dropout and dropout > 0:
        layers.append(nn.Dropout(float(dropout)))
      prev = int(h)

    out = nn.Linear(prev, out_dim)
    init_linear(out, gain=1.0)
    layers.append(out)

    self.net = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


# ----------------------------
# 3) LSTM for time series / sequence classification/regression
# ----------------------------

class LSTMEncoder(nn.Module):
  """
  LSTM encoder that produces a fixed-size representation per sequence.

  Input:
    x: (B, T, d_in)
    lengths: optional (B,) lengths for packed sequences (recommended for variable T)

  Output:
    rep: (B, rep_dim)
  """
  def __init__(
    self,
    d_in: int,
    hidden_size: int = 128,
    num_layers: int = 1,
    bidirectional: bool = False,
    dropout: float = 0.0,
    pooling: Literal["last", "mean"] = "last"
  ):
    super().__init__()
    if num_layers < 1:
      raise ValueError("num_layers must be >= 1")
    if pooling not in ("last", "mean"):
      raise ValueError("pooling must be 'last' or 'mean'")

    self.hidden_size = int(hidden_size)
    self.num_layers = int(num_layers)
    self.bidirectional = bool(bidirectional)
    self.pooling = pooling

    lstm_dropout = float(dropout) if num_layers > 1 else 0.0
    self.lstm = nn.LSTM(
      input_size=int(d_in),
      hidden_size=self.hidden_size,
      num_layers=self.num_layers,
      batch_first=True,
      bidirectional=self.bidirectional,
      dropout=lstm_dropout
    )

  @property
  def rep_dim(self) -> int:
    return self.hidden_size * (2 if self.bidirectional else 1)

  def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
    if lengths is not None:
      lengths = lengths.to(device=x.device).long()
      packed = nn.utils.rnn.pack_padded_sequence(
        x, lengths.cpu(), batch_first=True, enforce_sorted=False
      )
      packed_out, (h_n, _) = self.lstm(packed)
      out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
    else:
      out, (h_n, _) = self.lstm(x)

    if self.pooling == "last":
      # h_n shape: (num_layers * num_directions, B, hidden)
      # take last layer's hidden states, keeping both directions if bidirectional
      num_dirs = 2 if self.bidirectional else 1
      h_last = h_n[-num_dirs:]  # (num_dirs, B, hidden)
      rep = torch.cat([h_last[i] for i in range(h_last.shape[0])], dim=1)  # (B, hidden*num_dirs)
      return rep

    # mean pooling over time (mask if lengths provided)
    if lengths is None:
      return out.mean(dim=1)

    B, T, D = out.shape
    mask = torch.arange(T, device=out.device).unsqueeze(0).expand(B, T) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).type_as(out)
    summed = (out * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


class LSTMModel(nn.Module):
  """
  End-to-end LSTM model for:
  - regression:  outputs (B, d_out)
  - binary:      outputs (B, 1) logits
  - multiclass:  outputs (B, n_classes) logits

  Input:
    x: (B, T, d_in)
    lengths: optional (B,)
  """
  def __init__(
    self,
    d_in: int,
    task: TaskType,
    d_out: int = 1,
    n_classes: Optional[int] = None,
    hidden_size: int = 128,
    num_layers: int = 1,
    bidirectional: bool = False,
    dropout: float = 0.0,
    pooling: Literal["last", "mean"] = "last",
    head: Literal["linear", "mlp"] = "linear",
    head_hidden_sizes: Sequence[int] = (128,),
    head_dropout: float = 0.1
  ):
    super().__init__()
    self.task = task

    self.encoder = LSTMEncoder(
      d_in=d_in,
      hidden_size=hidden_size,
      num_layers=num_layers,
      bidirectional=bidirectional,
      dropout=dropout,
      pooling=pooling
    )

    if task == "multiclass":
      if n_classes is None or n_classes < 2:
        raise ValueError("For multiclass, provide n_classes >= 2")
      out_dim = int(n_classes)
    elif task == "binary":
      out_dim = 1
    elif task == "regression":
      out_dim = int(d_out)
    else:
      raise ValueError(f"Unknown task: {task}")

    rep_dim = self.encoder.rep_dim

    if head == "linear":
      self.head = nn.Linear(rep_dim, out_dim)
      init_linear(self.head)
    elif head == "mlp":
      # small MLP head
      self.head = MLP(
        d_in=rep_dim,
        task=("regression" if task == "regression" else ("binary" if task == "binary" else "multiclass")),
        d_out=out_dim if task == "regression" else 1,
        n_classes=out_dim if task == "multiclass" else None,
        hidden_sizes=head_hidden_sizes,
        dropout=head_dropout,
        activation="relu",
        layer_norm=False
      )
    else:
      raise ValueError("head must be 'linear' or 'mlp'")

  def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
    rep = self.encoder(x, lengths=lengths)
    return self.head(rep)


# ----------------------------
# 4) Matrix Factorization (embeddings)
# ----------------------------

class DotProductMF(nn.Module):
  """
  Matrix factorization scoring for *given embeddings*.

  Inputs:
    u: (B, d)
    v: (B, d)
  Output:
    score: (B, 1) (or (B,) if squeeze=True)

  Options:
    - optional learned bias terms from embeddings are not included here (pure dot product)
  """
  def __init__(self, squeeze: bool = False):
    super().__init__()
    self.squeeze = bool(squeeze)

  def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if u.shape != v.shape:
      raise ValueError(f"u and v must have same shape. Got {u.shape} vs {v.shape}")
    score = (u * v).sum(dim=1, keepdim=True)
    return score.squeeze(1) if self.squeeze else score


class MatrixFactorization(nn.Module):
  """
  Matrix factorization with embedding tables (user/item ids).

  Inputs:
    user_id: (B,) long
    item_id: (B,) long
  Output:
    score: (B, 1) (or (B,) if squeeze=True)

  Notes:
  - This is the classic MF model: dot(u, v) + user_bias + item_bias + global_bias
  - Useful for implicit or explicit feedback; loss depends on your setup.
  """
  def __init__(
    self,
    n_users: int,
    n_items: int,
    emb_dim: int = 64,
    use_bias: bool = True,
    squeeze: bool = False
  ):
    super().__init__()
    if n_users < 1 or n_items < 1:
      raise ValueError("n_users and n_items must be >= 1")

    self.user_emb = nn.Embedding(int(n_users), int(emb_dim))
    self.item_emb = nn.Embedding(int(n_items), int(emb_dim))
    nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.02)
    nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)

    self.use_bias = bool(use_bias)
    self.squeeze = bool(squeeze)

    if self.use_bias:
      self.user_bias = nn.Embedding(int(n_users), 1)
      self.item_bias = nn.Embedding(int(n_items), 1)
      nn.init.zeros_(self.user_bias.weight)
      nn.init.zeros_(self.item_bias.weight)
      self.global_bias = nn.Parameter(torch.zeros(1))

  def forward(self, user_id: torch.Tensor, item_id: torch.Tensor) -> torch.Tensor:
    if user_id.dtype != torch.long:
      user_id = user_id.long()
    if item_id.dtype != torch.long:
      item_id = item_id.long()

    u = self.user_emb(user_id)  # (B, d)
    v = self.item_emb(item_id)  # (B, d)
    score = (u * v).sum(dim=1, keepdim=True)  # (B, 1)

    if self.use_bias:
      score = score + self.user_bias(user_id) + self.item_bias(item_id) + self.global_bias

    return score.squeeze(1) if self.squeeze else score