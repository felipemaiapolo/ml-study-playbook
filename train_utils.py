# train_utils.py
# Generic training utilities for PyTorch models.
#
# Supports:
# - tasks: regression, binary classification, multiclass classification
# - optimizers: Adam, LBFGS (with strong Wolfe line search)
# - L2 regularization INCLUDED IN THE LOSS (ridge-style), configurable via TrainConfig.l2_reg
#
# Notes:
# - For classification, models should output LOGITS.
# - If optimizer=LBFGS, model + inputs + loss are converted to float64.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


TaskType = Literal["regression", "binary", "multiclass"]
OptType = Literal["adam", "lbfgs"]


@dataclass
class TrainConfig:
  task: TaskType = "regression"
  optimizer: OptType = "adam"
  device: str = "auto"  # "auto", "cpu", "cuda"

  epochs: int = 10
  lr: float = 1e-3

  # IMPORTANT: weight_decay is optimizer-level L2. If you set l2_reg too, you will double-count L2.
  # Prefer using l2_reg (loss-based) for consistency across Adam/LBFGS.
  weight_decay: float = 0.0

  # Loss-based L2 regularization (ridge): objective = data_loss + 0.5 * l2_reg * sum(||w||^2)
  l2_reg: float = 0.0
  l2_exclude_bias_norm: bool = True

  # Adam extras
  adam_betas: Tuple[float, float] = (0.9, 0.999)
  adam_eps: float = 1e-8

  # LBFGS extras
  lbfgs_lr: float = 1.0
  lbfgs_max_iter: int = 20
  lbfgs_history_size: int = 100
  lbfgs_line_search_fn: str = "strong_wolfe"  # line search

  # Loss options
  pos_weight: Optional[float] = None  # binary only (BCEWithLogitsLoss)
  label_smoothing: float = 0.0        # multiclass only

  # Training loop behavior
  grad_clip_norm: Optional[float] = None
  log_every: int = 50
  eval_every: int = 1

  # Early stopping on val_loss
  early_stopping_patience: Optional[int] = None

  # AMP (recommended only for Adam + CUDA)
  use_amp: bool = False

  # Misc
  max_eval_batches: Optional[int] = None


def resolve_device(device: str) -> torch.device:
  if device == "auto":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if device == "cuda":
    if not torch.cuda.is_available():
      raise RuntimeError("device=cuda requested but CUDA is not available.")
    return torch.device("cuda")
  if device == "cpu":
    return torch.device("cpu")
  raise ValueError(f"Unknown device: {device}")


def _move_to_device(x: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> Any:
  if torch.is_tensor(x):
    if dtype is not None and x.dtype.is_floating_point:
      return x.to(device=device, dtype=dtype)
    return x.to(device=device)
  if isinstance(x, dict):
    return {k: _move_to_device(v, device, dtype=dtype) for k, v in x.items()}
  if isinstance(x, (list, tuple)):
    return type(x)(_move_to_device(v, device, dtype=dtype) for v in x)
  return x


def _unpack_batch(batch: Any) -> Tuple[Any, Any]:
  if not isinstance(batch, (list, tuple)) or len(batch) != 2:
    raise ValueError("Expected batch to be (X, y).")
  return batch[0], batch[1]


def _infer_labels(y: torch.Tensor, task: TaskType) -> torch.Tensor:
  if task == "regression":
    if y.ndim == 1:
      return y.unsqueeze(1)
    return y

  if task == "binary":
    if y.ndim == 2 and y.shape[1] == 2:
      y = torch.argmax(y, dim=1)
    if y.ndim == 2 and y.shape[1] == 1:
      y = y.reshape(-1)
    return y.float()

  # multiclass
  if y.ndim == 2 and y.shape[1] > 1:
    y = torch.argmax(y, dim=1)
  if y.ndim == 2 and y.shape[1] == 1:
    y = y.reshape(-1)
  return y.long()


def make_loss_fn(cfg: TrainConfig) -> nn.Module:
  if cfg.task == "regression":
    return nn.MSELoss()

  if cfg.task == "binary":
    if cfg.pos_weight is None:
      return nn.BCEWithLogitsLoss()
    pos_w = torch.tensor([float(cfg.pos_weight)], dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pos_w)

  if cfg.task == "multiclass":
    try:
      return nn.CrossEntropyLoss(label_smoothing=float(cfg.label_smoothing))
    except TypeError:
      if cfg.label_smoothing != 0.0:
        raise RuntimeError("This PyTorch version doesn't support label_smoothing in CrossEntropyLoss.")
      return nn.CrossEntropyLoss()

  raise ValueError(f"Unknown task: {cfg.task}")


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
  if cfg.optimizer == "adam":
    return torch.optim.Adam(
      model.parameters(),
      lr=float(cfg.lr),
      weight_decay=float(cfg.weight_decay),
      betas=tuple(cfg.adam_betas),
      eps=float(cfg.adam_eps),
    )

  if cfg.optimizer == "lbfgs":
    return torch.optim.LBFGS(
      model.parameters(),
      lr=float(cfg.lbfgs_lr),
      max_iter=int(cfg.lbfgs_max_iter),
      history_size=int(cfg.lbfgs_history_size),
      line_search_fn=str(cfg.lbfgs_line_search_fn),
    )

  raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def _forward(model: nn.Module, X: Any) -> torch.Tensor:
  if isinstance(X, dict):
    out = model(**X)
  else:
    out = model(X)
  if not torch.is_tensor(out):
    raise ValueError("Model forward must return a torch.Tensor.")
  return out


def _l2_penalty(model: nn.Module, exclude_bias_norm: bool = True) -> torch.Tensor:
  """
  Returns sum(||w||^2) over parameters.
  If exclude_bias_norm=True, heuristically excludes:
  - biases (name endswith '.bias' or equals 'bias')
  - norm/bn parameters (if 'norm' or 'bn' in name)
  """
  total = None
  for name, p in model.named_parameters():
    if not p.requires_grad:
      continue
    if exclude_bias_norm:
      lname = name.lower()
      if lname.endswith("bias") or lname == "bias":
        continue
      if ("norm" in lname) or (".bn" in lname) or ("batchnorm" in lname):
        continue
      if p.ndim == 1:
        # catches many norm scales even if naming is odd
        continue
    term = (p ** 2).sum()
    total = term if total is None else (total + term)
  if total is None:
    # no parameters to regularize
    return torch.zeros((), device=next(model.parameters()).device)
  return total


def _compute_data_loss(
  logits_or_yhat: torch.Tensor,
  y: torch.Tensor,
  loss_fn: nn.Module,
  cfg: TrainConfig
) -> torch.Tensor:
  if cfg.task == "regression":
    if y.ndim == 1:
      y = y.unsqueeze(1)
    if logits_or_yhat.ndim == 1:
      logits_or_yhat = logits_or_yhat.unsqueeze(1)
    return loss_fn(logits_or_yhat, y)

  if cfg.task == "binary":
    if logits_or_yhat.ndim == 2 and logits_or_yhat.shape[1] == 1:
      logits_or_yhat = logits_or_yhat.reshape(-1)
    y = _infer_labels(y, "binary")
    return loss_fn(logits_or_yhat, y)

  y_lab = _infer_labels(y, "multiclass")
  return loss_fn(logits_or_yhat, y_lab)


def _compute_loss(
  model: nn.Module,
  logits_or_yhat: torch.Tensor,
  y: torch.Tensor,
  loss_fn: nn.Module,
  cfg: TrainConfig
) -> torch.Tensor:
  data_loss = _compute_data_loss(logits_or_yhat, y, loss_fn, cfg)

  if cfg.l2_reg and cfg.l2_reg > 0.0:
    l2 = _l2_penalty(model, exclude_bias_norm=cfg.l2_exclude_bias_norm)
    l2 = l2.to(dtype=data_loss.dtype)
    return data_loss + 0.5 * float(cfg.l2_reg) * l2

  return data_loss


@torch.no_grad()
def predict_on_loader(
  model: nn.Module,
  loader: DataLoader,
  cfg: TrainConfig
) -> Tuple[np.ndarray, np.ndarray]:
  model.eval()
  device = next(model.parameters()).device

  dtype = None
  if cfg.optimizer == "lbfgs":
    dtype = torch.float64

  preds: List[np.ndarray] = []
  ys: List[np.ndarray] = []

  for bi, batch in enumerate(loader):
    if cfg.max_eval_batches is not None and bi >= int(cfg.max_eval_batches):
      break

    X, y = _unpack_batch(batch)
    X = _move_to_device(X, device, dtype=dtype)
    y = _move_to_device(y, device, dtype=dtype)

    out = _forward(model, X)

    if cfg.task == "binary":
      if out.ndim == 1:
        out = out.unsqueeze(1)
      y_lab = _infer_labels(y, "binary").reshape(-1, 1)
      preds.append(out.detach().cpu().numpy())
      ys.append(y_lab.detach().cpu().numpy())
      continue

    if cfg.task == "multiclass":
      y_lab = _infer_labels(y, "multiclass").reshape(-1, 1)
      preds.append(out.detach().cpu().numpy())
      ys.append(y_lab.detach().cpu().numpy())
      continue

    if out.ndim == 1:
      out = out.unsqueeze(1)
    if y.ndim == 1:
      y = y.unsqueeze(1)
    preds.append(out.detach().cpu().numpy())
    ys.append(y.detach().cpu().numpy())

  preds_np = np.concatenate(preds, axis=0) if preds else np.zeros((0, 1), dtype=np.float64)
  y_np = np.concatenate(ys, axis=0) if ys else np.zeros((0, 1), dtype=np.float64)
  return preds_np, y_np


def train_one_epoch_adam(
  model: nn.Module,
  loader: DataLoader,
  optimizer: torch.optim.Optimizer,
  loss_fn: nn.Module,
  cfg: TrainConfig,
  scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> float:
  model.train()
  device = next(model.parameters()).device
  use_amp = bool(cfg.use_amp and device.type == "cuda")

  total_loss = 0.0
  n = 0

  for step, batch in enumerate(loader):
    X, y = _unpack_batch(batch)
    X = _move_to_device(X, device)
    y = _move_to_device(y, device)

    optimizer.zero_grad(set_to_none=True)

    if use_amp:
      with torch.cuda.amp.autocast():
        out = _forward(model, X)
        loss = _compute_loss(model, out, y, loss_fn, cfg)
      assert scaler is not None
      scaler.scale(loss).backward()
      if cfg.grad_clip_norm is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
      scaler.step(optimizer)
      scaler.update()
    else:
      out = _forward(model, X)
      loss = _compute_loss(model, out, y, loss_fn, cfg)
      loss.backward()
      if cfg.grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
      optimizer.step()

    bs = y.shape[0] if torch.is_tensor(y) else (len(y) if hasattr(y, "__len__") else 1)
    total_loss += float(loss.item()) * bs
    n += bs

    if cfg.log_every > 0 and (step + 1) % cfg.log_every == 0:
      print(f"  step {step+1:>5d} | train_loss {total_loss / max(1, n):.6f}")

  return total_loss / max(1, n)


def train_one_epoch_lbfgs(
  model: nn.Module,
  loader: DataLoader,
  optimizer: torch.optim.LBFGS,
  loss_fn: nn.Module,
  cfg: TrainConfig
) -> float:
  model.train()
  device = next(model.parameters()).device

  total_loss = 0.0
  n = 0

  for step, batch in enumerate(loader):
    X, y = _unpack_batch(batch)
    X = _move_to_device(X, device, dtype=torch.float64)
    y = _move_to_device(y, device, dtype=torch.float64)

    def closure() -> torch.Tensor:
      optimizer.zero_grad(set_to_none=True)
      out = _forward(model, X)
      loss = _compute_loss(model, out, y, loss_fn, cfg)
      loss.backward()
      return loss

    loss = optimizer.step(closure)
    loss_val = float(loss.item()) if torch.is_tensor(loss) else float(loss)

    bs = y.shape[0] if torch.is_tensor(y) else (len(y) if hasattr(y, "__len__") else 1)
    total_loss += loss_val * bs
    n += bs

    if cfg.log_every > 0 and (step + 1) % cfg.log_every == 0:
      print(f"  step {step+1:>5d} | train_loss {total_loss / max(1, n):.6f}")

  return total_loss / max(1, n)


@torch.no_grad()
def evaluate_loss(
  model: nn.Module,
  loader: DataLoader,
  loss_fn: nn.Module,
  cfg: TrainConfig
) -> float:
  model.eval()
  device = next(model.parameters()).device

  dtype = None
  if cfg.optimizer == "lbfgs":
    dtype = torch.float64

  total_loss = 0.0
  n = 0

  for bi, batch in enumerate(loader):
    if cfg.max_eval_batches is not None and bi >= int(cfg.max_eval_batches):
      break

    X, y = _unpack_batch(batch)
    X = _move_to_device(X, device, dtype=dtype)
    y = _move_to_device(y, device, dtype=dtype)

    out = _forward(model, X)
    loss = _compute_loss(model, out, y, loss_fn, cfg)

    bs = y.shape[0] if torch.is_tensor(y) else (len(y) if hasattr(y, "__len__") else 1)
    total_loss += float(loss.item()) * bs
    n += bs

  return total_loss / max(1, n)


def train_model(
  model: nn.Module,
  train_loader: DataLoader,
  val_loader: Optional[DataLoader],
  cfg: TrainConfig,
  metric_fn: Optional[Callable[[np.ndarray, np.ndarray], Dict[str, float]]] = None
) -> Dict[str, Any]:
  device = resolve_device(cfg.device)
  model = model.to(device)

  if cfg.optimizer == "lbfgs":
    model = model.double()
    cfg.use_amp = False

  loss_fn = make_loss_fn(cfg).to(device)
  if cfg.optimizer == "lbfgs":
    loss_fn = loss_fn.double()

  # If pos_weight is used, it must match dtype/device (especially for LBFGS double)
  if cfg.task == "binary" and cfg.pos_weight is not None:
    # BCEWithLogitsLoss keeps pos_weight as a tensor attribute
    if hasattr(loss_fn, "pos_weight") and loss_fn.pos_weight is not None:
      loss_fn.pos_weight = loss_fn.pos_weight.to(device=device, dtype=next(model.parameters()).dtype)

  optimizer = make_optimizer(model, cfg)

  scaler = None
  if cfg.optimizer == "adam" and cfg.use_amp and device.type == "cuda":
    scaler = torch.cuda.amp.GradScaler()

  history: List[Dict[str, Any]] = []
  best_state: Optional[Dict[str, torch.Tensor]] = None
  best_val_loss = float("inf")
  best_epoch = -1
  bad_epochs = 0

  for epoch in range(1, int(cfg.epochs) + 1):
    print(f"\nEpoch {epoch}/{cfg.epochs}")

    if cfg.optimizer == "adam":
      train_loss = train_one_epoch_adam(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        cfg=cfg,
        scaler=scaler
      )
    else:
      train_loss = train_one_epoch_lbfgs(
        model=model,
        loader=train_loader,
        optimizer=optimizer,  # type: ignore[arg-type]
        loss_fn=loss_fn,
        cfg=cfg
      )

    row: Dict[str, Any] = {"epoch": epoch, "train_loss": float(train_loss)}

    do_eval = (val_loader is not None) and (cfg.eval_every > 0) and (epoch % int(cfg.eval_every) == 0)
    if do_eval:
      val_loss = evaluate_loss(model, val_loader, loss_fn, cfg)
      row["val_loss"] = float(val_loss)

      if metric_fn is not None:
        preds_np, y_np = predict_on_loader(model, val_loader, cfg)
        try:
          extra = metric_fn(preds_np, y_np)
          for k, v in extra.items():
            row[k] = v
        except Exception as e:
          row["metric_error"] = str(e)

      if val_loss < best_val_loss:
        best_val_loss = float(val_loss)
        best_epoch = epoch
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        bad_epochs = 0
      else:
        bad_epochs += 1

      if cfg.early_stopping_patience is not None and bad_epochs >= int(cfg.early_stopping_patience):
        print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, best val_loss {best_val_loss:.6f})")
        history.append(row)
        break

    history.append(row)
    print(row)

  if best_state is None:
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = len(history)
    best_val_loss = float("nan")

  model.load_state_dict(best_state)

  return {
    "history": history,
    "best_state_dict": best_state,
    "best_epoch": best_epoch,
    "best_val_loss": best_val_loss,
  }
