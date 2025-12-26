# PyTorch Cheat Sheet (copy/paste friendly)

This is a practical “grab blocks and ship” guide. It’s opinionated: it favors patterns that don’t surprise you and don’t silently do the wrong thing.

---

## Contents
- [0. Setup + imports](#0-setup--imports)
- [1. Reproducibility](#1-reproducibility)
- [2. Device, dtype, and performance knobs](#2-device-dtype-and-performance-knobs)
- [3. Datasets + DataLoaders](#3-datasets--dataloaders)
- [4. Models](#4-models)
- [5. Losses](#5-losses)
- [6. Metrics + meters](#6-metrics--meters)
- [7. Optimizers + schedulers](#7-optimizers--schedulers)
- [8. Training loop (AMP, grad clip, accumulation)](#8-training-loop-amp-grad-clip-accumulation)
- [9. Evaluation + inference](#9-evaluation--inference)
- [10. Checkpointing + resume](#10-checkpointing--resume)
- [11. Early stopping](#11-early-stopping)
- [12. Debug playbook](#12-debug-playbook)
- [13. Common shape conventions](#13-common-shape-conventions)

---

## 0. Setup + imports

### requirements (minimal)
```txt
torch
numpy
````

### imports

```python
import os
import math
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
```

---

## 1. Reproducibility

If you care about repeatability, you must seed *and* tame CUDA nondeterminism (with a speed hit).

```python
def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
  os.environ["PYTHONHASHSEED"] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
  else:
    torch.backends.cudnn.benchmark = True
```

---

## 2. Device, dtype, and performance knobs

```python
def get_device() -> torch.device:
  return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

device = get_device()
print("device:", device)
```

### Fast(er) host->GPU transfers

Use `pin_memory=True` in DataLoader and `non_blocking=True` when moving tensors.

```python
def to_device(batch: Any, device: torch.device) -> Any:
  if torch.is_tensor(batch):
    return batch.to(device, non_blocking=True)
  if isinstance(batch, dict):
    return {k: to_device(v, device) for k, v in batch.items()}
  if isinstance(batch, (list, tuple)):
    t = [to_device(x, device) for x in batch]
    return type(batch)(t)
  return batch
```

### (Optional) torch.compile

Worth trying for stable models; if it breaks, disable it.

```python
def maybe_compile(model: nn.Module, enable: bool = False) -> nn.Module:
  if enable and hasattr(torch, "compile"):
    return torch.compile(model)
  return model
```

---

## 3. Datasets + DataLoaders

### 3.1 Map-style Dataset (most common)

You return one sample at a time.

```python
class NumpyDataset(Dataset):
  def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
    self.X = X.astype(np.float32)
    self.y = None if y is None else y.astype(np.float32)

  def __len__(self) -> int:
    return self.X.shape[0]

  def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    x = torch.from_numpy(self.X[idx])
    if self.y is None:
      return {"x": x}
    y = torch.from_numpy(self.y[idx])
    return {"x": x, "y": y}
```

### 3.2 Custom collate_fn (variable length)

Use when samples have different sizes (text sequences, variable #boxes, etc.).

```python
def pad_1d(seqs: List[torch.Tensor], pad_value: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
  lengths = torch.tensor([s.numel() for s in seqs], dtype=torch.long)
  max_len = int(lengths.max().item())
  out = seqs[0].new_full((len(seqs), max_len), fill_value=pad_value)
  for i, s in enumerate(seqs):
    out[i, :s.numel()] = s
  return out, lengths

def collate_text(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
  x_list = [b["x"] for b in batch]  # each x is (L,)
  y = torch.stack([b["y"] for b in batch]) if "y" in batch[0] else None
  x, lengths = pad_1d(x_list, pad_value=0)
  out = {"x": x, "lengths": lengths}
  if y is not None:
    out["y"] = y
  return out
```

### 3.3 DataLoader template

```python
def make_loader(
  ds: Dataset,
  batch_size: int,
  shuffle: bool,
  num_workers: int = 4,
  collate_fn=None
) -> DataLoader:
  return DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    drop_last=shuffle,
    collate_fn=collate_fn
  )
```

---

## 4. Models

### 4.1 MLP (tabular / embeddings already handled)

```python
class MLP(nn.Module):
  def __init__(self, in_dim: int, hidden: List[int], out_dim: int, p_drop: float = 0.0):
    super().__init__()
    layers = []
    prev = in_dim
    for h in hidden:
      layers += [
        nn.Linear(prev, h),
        nn.ReLU(inplace=True),
        nn.Dropout(p_drop)
      ]
      prev = h
    layers.append(nn.Linear(prev, out_dim))
    self.net = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)
```

### 4.2 CNN for images (N, C, H, W)

```python
class SmallCNN(nn.Module):
  def __init__(self, in_ch: int = 3, num_classes: int = 10):
    super().__init__()
    self.features = nn.Sequential(
      nn.Conv2d(in_ch, 32, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 64, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2)
    )
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(64 * 8 * 8, 256),  # adjust if your input size differs
      nn.ReLU(inplace=True),
      nn.Linear(256, num_classes)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    return self.classifier(x)
```

### 4.3 LSTM for padded sequences

`x`: `(N, L)` token ids, `lengths`: `(N,)`

```python
class LSTMClassifier(nn.Module):
  def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, num_classes: int):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
    self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
    self.head = nn.Linear(2 * hidden_dim, num_classes)

  def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    e = self.emb(x)  # (N, L, D)
    packed = nn.utils.rnn.pack_padded_sequence(
      e, lengths.cpu(), batch_first=True, enforce_sorted=False
    )
    _, (h, _) = self.lstm(packed)
    h = torch.cat([h[-2], h[-1]], dim=1)  # (N, 2H)
    return self.head(h)
```

### 4.4 Simple Transformer Encoder (classification)

Good enough for many sequence tasks.

```python
class TransformerClassifier(nn.Module):
  def __init__(
    self,
    vocab_size: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    num_classes: int,
    max_len: int = 512
  ):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
    self.pos = nn.Embedding(max_len, d_model)
    enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
    self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
    self.head = nn.Linear(d_model, num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (N, L)
    N, L = x.shape
    pos = torch.arange(L, device=x.device).unsqueeze(0).expand(N, L)
    h = self.emb(x) + self.pos(pos)

    key_padding_mask = (x == 0)  # True where padding
    h = self.enc(h, src_key_padding_mask=key_padding_mask)

    # mean pool over non-pad
    mask = (~key_padding_mask).float().unsqueeze(-1)  # (N, L, 1)
    h = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return self.head(h)
```

### 4.5 Tabular with categorical embeddings (+ numeric)

```python
class TabularModel(nn.Module):
  def __init__(
    self,
    num_numeric: int,
    cat_cardinalities: List[int],
    cat_emb_dim: int = 16,
    hidden: List[int] = [256, 128],
    out_dim: int = 1
  ):
    super().__init__()
    self.cat_embs = nn.ModuleList([
      nn.Embedding(card, cat_emb_dim) for card in cat_cardinalities
    ])
    in_dim = num_numeric + cat_emb_dim * len(cat_cardinalities)
    self.mlp = MLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim, p_drop=0.1)

  def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
    # x_num: (N, num_numeric)
    # x_cat: (N, num_cats) with integer codes
    embs = []
    for j, emb in enumerate(self.cat_embs):
      embs.append(emb(x_cat[:, j]))
    x = torch.cat([x_num] + embs, dim=1)
    return self.mlp(x)
```

---

## 5. Losses

### 5.1 Regression

```python
loss_fn = nn.MSELoss()      # L2
# loss_fn = nn.L1Loss()     # L1
# loss_fn = nn.HuberLoss(delta=1.0)
```

### 5.2 Multi-class classification (logits)

* Model outputs raw logits `(N, C)`
* Targets are class indices `(N,)` with dtype `long`

```python
loss_fn = nn.CrossEntropyLoss()
```

### 5.3 Binary classification (logits)

* Model outputs logits `(N,)` or `(N, 1)`
* Targets are floats in `{0,1}`

```python
loss_fn = nn.BCEWithLogitsLoss()
```

### 5.4 Multi-label classification (logits)

* logits `(N, C)`, targets float `(N, C)` in `{0,1}`

```python
loss_fn = nn.BCEWithLogitsLoss()
```

---

## 6. Metrics + meters

### 6.1 AverageMeter (the one you actually want)

```python
class AverageMeter:
  def __init__(self):
    self.reset()

  def reset(self):
    self.sum = 0.0
    self.count = 0

  def update(self, val: float, n: int = 1):
    self.sum += float(val) * n
    self.count += int(n)

  @property
  def avg(self) -> float:
    if self.count == 0:
      return 0.0
    return self.sum / self.count
```

### 6.2 Accuracy (top-1)

```python
@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, y: torch.Tensor) -> float:
  pred = logits.argmax(dim=1)
  return (pred == y).float().mean().item()
```

### 6.3 R2 for regression

```python
@torch.no_grad()
def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
  y_true = y_true.float().view(-1)
  y_pred = y_pred.float().view(-1)
  ss_res = torch.sum((y_true - y_pred) ** 2)
  ss_tot = torch.sum((y_true - y_true.mean()) ** 2).clamp_min(1e-12)
  return (1.0 - ss_res / ss_tot).item()
```

---

## 7. Optimizers + schedulers

### 7.1 Optimizer defaults that won’t embarrass you

```python
def make_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
  return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
```

### 7.2 Common schedulers

Step per *epoch*:

```python
def make_scheduler_epoch(optimizer: torch.optim.Optimizer, max_epochs: int):
  return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
```

Step per *iteration* (useful with warmup):

```python
def make_scheduler_iter(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int = 0):
  def lr_lambda(step: int):
    if warmup_steps > 0 and step < warmup_steps:
      return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
  return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
```

---

## 8. Training loop (AMP, grad clip, accumulation)

This is the block you’ll reuse the most.

### 8.1 One epoch: train

```python
def train_one_epoch(
  model: nn.Module,
  loader: DataLoader,
  optimizer: torch.optim.Optimizer,
  loss_fn,
  device: torch.device,
  scaler: Optional[torch.cuda.amp.GradScaler] = None,
  scheduler=None,
  grad_clip: float = 0.0,
  accumulate_steps: int = 1
) -> Dict[str, float]:
  model.train()
  loss_meter = AverageMeter()

  optimizer.zero_grad(set_to_none=True)

  for step, batch in enumerate(loader):
    batch = to_device(batch, device)

    with torch.cuda.amp.autocast(enabled=(scaler is not None)):
      logits = model(batch["x"]) if "lengths" not in batch else model(batch["x"], batch["lengths"])
      y = batch["y"]
      loss = loss_fn(logits, y)
      loss = loss / accumulate_steps

    if scaler is None:
      loss.backward()
      if grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
      if (step + 1) % accumulate_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    else:
      scaler.scale(loss).backward()
      if (step + 1) % accumulate_steps == 0:
        if grad_clip > 0:
          scaler.unscale_(optimizer)
          nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    bs = batch["x"].shape[0]
    loss_meter.update(loss.item() * accumulate_steps, n=bs)

    if scheduler is not None:
      # If this scheduler is per-iteration, step here.
      # If it's per-epoch, step outside this function.
      if isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
        scheduler.step()

  return {"loss": loss_meter.avg}
```

### 8.2 One epoch: eval

```python
@torch.no_grad()
def eval_one_epoch(
  model: nn.Module,
  loader: DataLoader,
  loss_fn,
  device: torch.device
) -> Dict[str, float]:
  model.eval()
  loss_meter = AverageMeter()

  for batch in loader:
    batch = to_device(batch, device)
    logits = model(batch["x"]) if "lengths" not in batch else model(batch["x"], batch["lengths"])
    y = batch["y"]
    loss = loss_fn(logits, y)
    bs = batch["x"].shape[0]
    loss_meter.update(loss.item(), n=bs)

  return {"loss": loss_meter.avg}
```

### 8.3 Full fit loop (best checkpoint + scheduler per epoch)

```python
def fit(
  model: nn.Module,
  train_loader: DataLoader,
  val_loader: DataLoader,
  optimizer: torch.optim.Optimizer,
  loss_fn,
  device: torch.device,
  epochs: int,
  scheduler_epoch=None,
  amp: bool = True,
  grad_clip: float = 0.0,
  accumulate_steps: int = 1
) -> Dict[str, Any]:
  scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))
  best_val = float("inf")
  best_state = None

  for epoch in range(1, epochs + 1):
    t0 = time.time()

    tr = train_one_epoch(
      model=model,
      loader=train_loader,
      optimizer=optimizer,
      loss_fn=loss_fn,
      device=device,
      scaler=scaler if scaler.is_enabled() else None,
      scheduler=None,
      grad_clip=grad_clip,
      accumulate_steps=accumulate_steps
    )

    va = eval_one_epoch(
      model=model,
      loader=val_loader,
      loss_fn=loss_fn,
      device=device
    )

    if scheduler_epoch is not None:
      scheduler_epoch.step()

    dt = time.time() - t0
    lr = optimizer.param_groups[0]["lr"]
    print(f"epoch {epoch:03d} | lr {lr:.3e} | train {tr['loss']:.4f} | val {va['loss']:.4f} | {dt:.1f}s")

    if va["loss"] < best_val:
      best_val = va["loss"]
      best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

  if best_state is not None:
    model.load_state_dict(best_state)

  return {"best_val_loss": best_val}
```

---

## 9. Evaluation + inference

### 9.1 Batched prediction (no gradients, faster)

```python
@torch.no_grad()
def predict(
  model: nn.Module,
  loader: DataLoader,
  device: torch.device
) -> torch.Tensor:
  model.eval()
  preds = []

  with torch.inference_mode():
    for batch in loader:
      batch = to_device(batch, device)
      out = model(batch["x"]) if "lengths" not in batch else model(batch["x"], batch["lengths"])
      preds.append(out.detach().cpu())

  return torch.cat(preds, dim=0)
```

---

## 10. Checkpointing + resume

### 10.1 Save / load

Save *everything you need to resume*: model, optimizer, epoch, scaler, and whatever metrics you care about.

```python
def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  torch.save(payload, path)

def load_checkpoint(path: str, map_location="cpu") -> Dict[str, Any]:
  return torch.load(path, map_location=map_location)
```

### 10.2 Example usage

```python
# save
ckpt = {
  "model": model.state_dict(),
  "optimizer": optimizer.state_dict(),
  "epoch": epoch,
  "best_val_loss": best_val
}
save_checkpoint("checkpoints/last.pt", ckpt)

# load
ckpt = load_checkpoint("checkpoints/last.pt", map_location=device)
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
start_epoch = ckpt["epoch"] + 1
```

---

## 11. Early stopping

```python
class EarlyStopping:
  def __init__(self, patience: int = 10, min_delta: float = 0.0):
    self.patience = patience
    self.min_delta = min_delta
    self.best = float("inf")
    self.bad = 0

  def step(self, metric: float) -> bool:
    # returns True if should stop
    if metric < self.best - self.min_delta:
      self.best = metric
      self.bad = 0
      return False
    self.bad += 1
    return self.bad >= self.patience
```

Usage inside your fit:

```python
early = EarlyStopping(patience=5, min_delta=1e-4)
# after computing val_loss:
if early.step(val_loss):
  print("early stopping")
  break
```

---

## 12. Debug playbook

These catch 80% of “why is this not learning?” problems.

### 12.1 Overfit one batch

If you can’t drive loss near zero on a tiny batch, your pipeline is broken (shapes, labels, loss, model output).

```python
def overfit_one_batch(model, batch, loss_fn, device, steps: int = 200, lr: float = 1e-2):
  model = model.to(device)
  model.train()
  opt = torch.optim.AdamW(model.parameters(), lr=lr)
  batch = to_device(batch, device)

  for i in range(steps):
    opt.zero_grad(set_to_none=True)
    out = model(batch["x"]) if "lengths" not in batch else model(batch["x"], batch["lengths"])
    loss = loss_fn(out, batch["y"])
    loss.backward()
    opt.step()
    if (i + 1) % 20 == 0:
      print(i + 1, float(loss.item()))
```

### 12.2 Detect NaNs early

```python
torch.autograd.set_detect_anomaly(True)
```

### 12.3 Gradient sanity check

```python
def grad_norm(model: nn.Module) -> float:
  total = 0.0
  for p in model.parameters():
    if p.grad is None:
      continue
    total += p.grad.detach().data.norm(2).item() ** 2
  return total ** 0.5
```

---

## 13. Common shape conventions

### Regression

* Model output: `(N,)` or `(N, 1)`
* Target: same shape, float32
* Loss: `MSELoss`, `HuberLoss`, etc.

### Multi-class classification

* Model output: logits `(N, C)`
* Target: class index `(N,)`, dtype `torch.long`
* Loss: `CrossEntropyLoss` (expects logits, not softmax)

### Binary classification

* Model output: logits `(N,)` or `(N, 1)`
* Target: `(N,)` float in `{0,1}`
* Loss: `BCEWithLogitsLoss` (expects logits, not sigmoid)

---

## Minimal end-to-end example (tabular regression)

```python
seed_everything(42)

# fake data
N, D = 4096, 20
X = np.random.randn(N, D).astype(np.float32)
w = np.random.randn(D).astype(np.float32)
y = (X @ w + 0.1 * np.random.randn(N)).astype(np.float32)

# split
idx = np.arange(N)
np.random.shuffle(idx)
tr_idx, va_idx = idx[:3000], idx[3000:]

train_ds = NumpyDataset(X[tr_idx], y[tr_idx])
val_ds = NumpyDataset(X[va_idx], y[va_idx])

train_loader = make_loader(train_ds, batch_size=128, shuffle=True, num_workers=2)
val_loader = make_loader(val_ds, batch_size=256, shuffle=False, num_workers=2)

device = get_device()
model = MLP(in_dim=D, hidden=[256, 128], out_dim=1, p_drop=0.1).to(device)
model = maybe_compile(model, enable=False)

loss_fn = nn.MSELoss()
optimizer = make_optimizer(model, lr=1e-3, weight_decay=1e-4)
scheduler = make_scheduler_epoch(optimizer, max_epochs=10)

fit(
  model=model,
  train_loader=train_loader,
  val_loader=val_loader,
  optimizer=optimizer,
  loss_fn=loss_fn,
  device=device,
  epochs=10,
  scheduler_epoch=scheduler,
  amp=True,
  grad_clip=1.0,
  accumulate_steps=1
)
```

---

## Last blunt notes (read once)

* If metrics don’t move, **overfit one batch**. If that fails, stop training longer — fix the bug.
* Don’t apply `softmax` before `CrossEntropyLoss`. Don’t apply `sigmoid` before `BCEWithLogitsLoss`.
* Always call `model.train()` for training and `model.eval()` for eval/inference.
* Use `optimizer.zero_grad(set_to_none=True)` (less memory traffic, fewer surprises).
* If you use padding, you must mask it (or pack sequences). Otherwise your model learns garbage.


