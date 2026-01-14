# toy_data_utils.py
# Synthetic data generators for quick smoke tests / debugging.
#
# Design:
# - All functions return numpy arrays by default.
# - Shapes are 2D for X and y where applicable (n, d_x) and (n, d_y).
# - Deterministic via numpy Generator (seed).

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np


def _rng(seed: int) -> np.random.Generator:
  return np.random.default_rng(int(seed))


def _ensure_2d_y(y: np.ndarray) -> np.ndarray:
  if y.ndim == 1:
    return y.reshape(-1, 1)
  if y.ndim != 2:
    raise ValueError(f"y must be 1D or 2D, got shape {y.shape}")
  return y


# ----------------------------
# Regression: linear / nonlinear, optional multi-output
# ----------------------------

def make_regression_data(
  n: int = 10000,
  d_x: int = 20,
  d_y: int = 1,
  noise_std: float = 0.5,
  nonlinear: bool = False,
  seed: int = 0,
  x_dist: str = "normal"  # "normal" or "uniform"
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
  """
  Returns:
    X: (n, d_x)
    y: (n, d_y)
    info: dict with true parameters (W, b) and maybe transform details
  """
  rng = _rng(seed)

  if x_dist == "normal":
    X = rng.standard_normal((n, d_x)).astype(np.float32)
  elif x_dist == "uniform":
    X = rng.uniform(-1.0, 1.0, size=(n, d_x)).astype(np.float32)
  else:
    raise ValueError("x_dist must be 'normal' or 'uniform'")

  W = rng.standard_normal((d_x, d_y)).astype(np.float32)
  b = rng.standard_normal((d_y,)).astype(np.float32)

  y_lin = X @ W + b  # (n, d_y)

  if nonlinear:
    # simple controlled nonlinearity: add sin + quadratic terms
    y = y_lin + 0.5 * np.sin(y_lin) + 0.1 * (y_lin ** 2)
  else:
    y = y_lin

  y = y + noise_std * rng.standard_normal((n, d_y)).astype(np.float32)

  info = {"W": W, "b": b}
  return X, y.astype(np.float32), info


# ----------------------------
# Classification: binary or multiclass
# ----------------------------

def make_classification_data(
  n: int = 10000,
  d_x: int = 20,
  n_classes: int = 2,
  noise_std: float = 1.0,
  separability: float = 1.0,
  seed: int = 0,
  return_probs: bool = False,
  one_hot_y: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
  """
  Synthetic classification via a random linear score model:
    logits = XW + b + noise

  For binary:
    y in {0,1}
    probs = sigmoid(logit)

  For multiclass:
    y in {0..C-1}
    probs = softmax(logits)

  Returns:
    X: (n, d_x)
    y: (n, 1) labels OR (n, C) one-hot if one_hot_y=True
    info: dict with W, b, and (optional) probs
  """
  if n_classes < 2:
    raise ValueError("n_classes must be >= 2")

  rng = _rng(seed)
  X = rng.standard_normal((n, d_x)).astype(np.float32)

  W = (separability * rng.standard_normal((d_x, n_classes))).astype(np.float32)
  b = (separability * rng.standard_normal((n_classes,))).astype(np.float32)

  logits = X @ W + b
  logits = logits + noise_std * rng.standard_normal(logits.shape).astype(np.float32)

  if n_classes == 2:
    # use logit for class 1
    z = logits[:, 1] - logits[:, 0]
    probs_pos = 1.0 / (1.0 + np.exp(-z.astype(np.float64)))
    y_labels = (rng.random(n) < probs_pos).astype(np.int64)  # stochastic labels

    if one_hot_y:
      y = np.zeros((n, 2), dtype=np.int64)
      y[np.arange(n), y_labels] = 1
    else:
      y = y_labels.reshape(-1, 1)

    info = {"W": W, "b": b}
    if return_probs:
      # return as (n,2) probs
      probs = np.stack([1.0 - probs_pos, probs_pos], axis=1).astype(np.float32)
      info["probs"] = probs
    return X, y, info

  # multiclass
  # stable softmax
  lg = logits.astype(np.float64)
  lg = lg - np.max(lg, axis=1, keepdims=True)
  exps = np.exp(lg)
  probs = exps / np.sum(exps, axis=1, keepdims=True)

  # sample labels from categorical(probs)
  y_labels = np.array([rng.choice(n_classes, p=probs[i]) for i in range(n)], dtype=np.int64)

  if one_hot_y:
    y = np.zeros((n, n_classes), dtype=np.int64)
    y[np.arange(n), y_labels] = 1
  else:
    y = y_labels.reshape(-1, 1)

  info = {"W": W, "b": b}
  if return_probs:
    info["probs"] = probs.astype(np.float32)
  return X, y, info


# ----------------------------
# Matrix completion / factorization
# ----------------------------

def make_matrix_completion_data(
  n_users: int = 1000,
  n_items: int = 800,
  rank: int = 20,
  n_obs: int = 200000,
  noise_std: float = 0.1,
  seed: int = 0,
  implicit: bool = False,
  implicit_thresh: float = 0.0,
  return_full_matrix: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
  """
  Generate a low-rank matrix R = U V^T + noise and sample observed entries.

  Returns:
    user_id: (n_obs, 1) int64
    item_id: (n_obs, 1) int64
    y: (n_obs, 1) float32 (ratings) OR int64 (implicit labels) if implicit=True
    info: dict containing U, V, and (optional) full_matrix

  Notes:
    - Observations are sampled uniformly over user-item pairs (with replacement).
    - If you want unique pairs, you can post-process, but for training MF this is fine.
  """
  rng = _rng(seed)

  U = rng.standard_normal((n_users, rank)).astype(np.float32)
  V = rng.standard_normal((n_items, rank)).astype(np.float32)

  user_id = rng.integers(0, n_users, size=(n_obs,), dtype=np.int64)
  item_id = rng.integers(0, n_items, size=(n_obs,), dtype=np.int64)

  ratings = (U[user_id] * V[item_id]).sum(axis=1)
  ratings = ratings + noise_std * rng.standard_normal((n_obs,)).astype(np.float32)

  if implicit:
    y = (ratings > implicit_thresh).astype(np.int64).reshape(-1, 1)
  else:
    y = ratings.astype(np.float32).reshape(-1, 1)

  info: Dict[str, np.ndarray] = {"U": U, "V": V}

  if return_full_matrix:
    full = (U @ V.T).astype(np.float32)
    info["full_matrix"] = full

  return user_id.reshape(-1, 1), item_id.reshape(-1, 1), y, info
