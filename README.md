```markdown
# PyTorch Applied ML — Study Repo Cheat Sheet

A practical, copy-pasteable set of **PyTorch patterns** for solving “given a dataset → build a model → train → evaluate → iterate” problems fast.

---

## Environment quickstart

### `requirements.txt` (minimal)
```txt
torch
torchvision
numpy
pandas
scikit-learn
tqdm
````

### Reproducibility seed

```python
# src/utils/seed.py
import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)

  if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## Core training skeleton (most common use)

This is the fastest “works everywhere” template: **AMP**, **grad clipping**, **logging**, **early stopping**, **checkpoint best**.

```python
# src/train.py
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.seed import seed_everything
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.meters import AverageMeter

@dataclass
class TrainConfig:
  seed: int = 42
  device: str = "cuda"
  epochs: int = 10
  batch_size: int = 256
  lr: float = 3e-4
  weight_decay: float = 1e-2
  grad_clip: float = 1.0
  amp: bool = True
  num_workers: int = 4
  ckpt_dir: str = "checkpoints"
  early_stop_patience: int = 5

def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device):
  model.train()
  loss_meter = AverageMeter()

  pbar = tqdm(loader, desc="train", leave=False)
  for batch in pbar:
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device.split(":")[0], enabled=scaler is not None):
      logits = model(x)
      loss = loss_fn(logits, y)

    if scaler is not None:
      scaler.scale(loss).backward()
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      scaler.step(optimizer)
      scaler.update()
    else:
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()

    loss_meter.update(loss.item(), n=x.size(0))
    pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

  return {"loss": loss_meter.avg}

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, metric_fn=None):
  model.eval()
  loss_meter = AverageMeter()
  metric_meter = AverageMeter()

  for batch in tqdm(loader, desc="eval", leave=False):
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    logits = model(x)
    loss = loss_fn(logits, y)
    loss_meter.update(loss.item(), n=x.size(0))

    if metric_fn is not None:
      m = metric_fn(logits, y)
      metric_meter.update(float(m), n=x.size(0))

  out = {"loss": loss_meter.avg}
  if metric_fn is not None:
    out["metric"] = metric_meter.avg
  return out

def main():
  cfg = TrainConfig()
  seed_everything(cfg.seed)

  device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
  Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)

  # TODO: plug your dataset/model
  from data.datasets import ToyDataset
  from models.mlp import MLP

  train_ds = ToyDataset(split="train")
  val_ds = ToyDataset(split="val")

  train_loader = DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True,
    drop_last=True,
  )
  val_loader = DataLoader(
    val_ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
  )

  model = MLP(in_dim=train_ds.in_dim, hidden=[256, 256], out_dim=train_ds.out_dim).to(device)

  # Choose loss + metric per task (see sections below)
  loss_fn = nn.CrossEntropyLoss()
  metric_fn = accuracy

  optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
  scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

  best = -math.inf
  bad_epochs = 0

  for epoch in range(cfg.epochs):
    tr = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, device)
    va = evaluate(model, val_loader, loss_fn, device, metric_fn=metric_fn)

    score = va.get("metric", -va["loss"])  # higher is better
    print(f"epoch {epoch:03d} | train loss {tr['loss']:.4f} | val {va}")

    is_best = score > best
    if is_best:
      best = score
      bad_epochs = 0
    else:
      bad_epochs += 1

    save_checkpoint(
      {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best": best,
      },
      Path(cfg.ckpt_dir) / ("best.pt" if is_best else "last.pt")
    )

    if bad_epochs >= cfg.early_stop_patience:
      print("Early stopping.")
      break

def accuracy(logits, y):
  preds = logits.argmax(dim=-1)
  return (preds == y).float().mean()

if __name__ == "__main__":
  main()
```

---

## Checkpoint utilities (safe + simple)

```python
# src/utils/checkpoint.py
import torch

def save_checkpoint(state: dict, path) -> None:
  torch.save(state, path)

def load_checkpoint(path, map_location="cpu") -> dict:
  return torch.load(path, map_location=map_location)
```

---

## Dataset + DataLoader patterns

### 1) Tabular dataset from CSV (classification or regression)

```python
# src/data/datasets.py
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

@dataclass
class TabularSpec:
  feature_cols: list[str]
  target_col: str
  task: str  # "clf" or "reg"

class CSVDataset(Dataset):
  def __init__(self, csv_path: str, spec: TabularSpec):
    df = pd.read_csv(csv_path)

    X = df[spec.feature_cols].to_numpy(dtype=np.float32)
    y = df[spec.target_col].to_numpy()

    if spec.task == "clf":
      y = y.astype(np.int64)
    else:
      y = y.astype(np.float32)

    self.X = torch.from_numpy(X)
    self.y = torch.from_numpy(y)

    self.in_dim = self.X.shape[1]
    self.out_dim = int(np.max(y) + 1) if spec.task == "clf" else 1
    self.task = spec.task

  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, idx):
    x = self.X[idx]
    y = self.y[idx]
    if self.task == "reg":
      y = y.view(1)
    return x, y
```

### 2) Image dataset (folder or custom)

Use `torchvision.datasets.ImageFolder` whenever possible.

```python
from torchvision import datasets, transforms

tfm = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
])

train_ds = datasets.ImageFolder("data/train", transform=tfm)
val_ds = datasets.ImageFolder("data/val", transform=tfm)
```

### 3) Variable-length sequences + padding collate

```python
# src/data/collate.py
import torch

def pad_collate(batch, pad_value: int = 0):
  # batch: list[(seq_tensor[L], label)]
  xs, ys = zip(*batch)
  lengths = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
  max_len = int(lengths.max().item())

  x_padded = []
  for x in xs:
    pad_len = max_len - x.size(0)
    if pad_len > 0:
      x = torch.cat([x, x.new_full((pad_len, *x.shape[1:]), pad_value)], dim=0)
    x_padded.append(x)

  x_padded = torch.stack(x_padded, dim=0)  # [B, T, ...]
  y = torch.tensor(ys)
  return x_padded, y, lengths
```

---

## Model templates you can adapt quickly

### MLP (tabular baseline)

```python
# src/models/mlp.py
import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, in_dim: int, hidden: list[int], out_dim: int, dropout: float = 0.0):
    super().__init__()
    layers = []
    d = in_dim
    for h in hidden:
      layers += [
        nn.Linear(d, h),
        nn.ReLU(),
        nn.Dropout(dropout),
      ]
      d = h
    layers.append(nn.Linear(d, out_dim))
    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x)
```

### Simple CNN (images baseline)

```python
# src/models/cnn.py
import torch.nn as nn

class SmallCNN(nn.Module):
  def __init__(self, num_classes: int):
    super().__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2),
    )
    self.head = nn.Sequential(
      nn.Flatten(),
      nn.Linear(64 * 56 * 56, 256),  # if input was 224x224
      nn.ReLU(),
      nn.Linear(256, num_classes),
    )

  def forward(self, x):
    x = self.features(x)
    return self.head(x)
```

### Transformer encoder (sequence baseline)

Good for tokens or time series. Shape: `x: [B, T, d_model]`.

```python
# src/models/transformer.py
import torch
import torch.nn as nn

class TransformerEncoderModel(nn.Module):
  def __init__(self, d_in: int, d_model: int, nhead: int, num_layers: int, d_ff: int, out_dim: int, dropout: float = 0.1):
    super().__init__()
    self.proj = nn.Linear(d_in, d_model)
    enc_layer = nn.TransformerEncoderLayer(
      d_model=d_model,
      nhead=nhead,
      dim_feedforward=d_ff,
      dropout=dropout,
      batch_first=True,
      activation="gelu",
      norm_first=True,
    )
    self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
    self.head = nn.Linear(d_model, out_dim)

  def forward(self, x, key_padding_mask=None):
    # key_padding_mask: [B, T] True for PAD positions
    h = self.proj(x)
    h = self.encoder(h, src_key_padding_mask=key_padding_mask)
    # pool: take CLS-like first token or mean
    pooled = h.mean(dim=1)
    return self.head(pooled)
```

---

## Losses (what to reach for)

### Classification

* **Softmax multi-class**: Cross entropy
  [
  \mathcal{L} = -\log p_{y}, \quad p=\mathrm{softmax}(z)
  ]

```python
loss_fn = nn.CrossEntropyLoss()
# logits: [B, C], y: [B] long
```

* **Binary / multi-label**: BCE with logits
  [
  \mathcal{L} = -y\log \sigma(z) - (1-y)\log(1-\sigma(z))
  ]

```python
loss_fn = nn.BCEWithLogitsLoss()
# logits: [B] or [B, C], y: same shape float in {0,1}
```

* **Label smoothing** (quick helper)

```python
import torch
import torch.nn.functional as F

def cross_entropy_label_smoothing(logits, y, eps: float = 0.1):
  # logits: [B, C], y: [B]
  n_classes = logits.size(-1)
  log_probs = F.log_softmax(logits, dim=-1)
  nll = -log_probs.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
  smooth = -log_probs.mean(dim=-1)
  return ((1 - eps) * nll + eps * smooth).mean()
```

### Regression

* **MSE**: (\mathcal{L}=|y-\hat y|_2^2)

```python
loss_fn = nn.MSELoss()
```

* **Huber / SmoothL1** (robust to outliers):

```python
loss_fn = nn.SmoothL1Loss(beta=1.0)
```

### Metric learning (common “bonus” problems)

* **Triplet loss**: enforce (d(a,p)+m < d(a,n))

```python
loss_fn = nn.TripletMarginLoss(margin=0.2, p=2)
```

---

## Metrics (fast, reliable snippets)

### Accuracy / Top-k

```python
import torch

@torch.no_grad()
def accuracy_topk(logits, y, k: int = 5):
  # logits: [B, C], y: [B]
  topk = logits.topk(k, dim=-1).indices  # [B, k]
  correct = topk.eq(y.unsqueeze(-1)).any(dim=-1).float()
  return correct.mean()
```

### F1 (binary, thresholded)

```python
@torch.no_grad()
def f1_binary_from_logits(logits, y, threshold: float = 0.5, eps: float = 1e-12):
  # logits, y: [B] (y in {0,1})
  probs = torch.sigmoid(logits)
  pred = (probs >= threshold).float()
  tp = (pred * y).sum()
  fp = (pred * (1 - y)).sum()
  fn = ((1 - pred) * y).sum()
  precision = tp / (tp + fp + eps)
  recall = tp / (tp + fn + eps)
  return 2 * precision * recall / (precision + recall + eps)
```

### RMSE / MAE

```python
import torch

@torch.no_grad()
def rmse(pred, y):
  return torch.sqrt(torch.mean((pred - y) ** 2))

@torch.no_grad()
def mae(pred, y):
  return torch.mean(torch.abs(pred - y))
```

### AUC (use sklearn when allowed)

```python
# caution: requires moving to CPU
from sklearn.metrics import roc_auc_score

def auc_from_logits(logits, y):
  probs = torch.sigmoid(logits).detach().cpu().numpy()
  y_np = y.detach().cpu().numpy()
  return roc_auc_score(y_np, probs)
```

---

## Optimizers + schedulers (battle-tested picks)

### AdamW (default)

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
```

### Cosine schedule with warmup (simple)

```python
import math

def lr_lambda_cosine_warmup(step, warmup_steps, total_steps):
  if step < warmup_steps:
    return (step + 1) / max(1, warmup_steps)
  progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
  return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(
  optimizer,
  lr_lambda=lambda s: lr_lambda_cosine_warmup(s, warmup_steps=500, total_steps=10_000),
)
```

### ReduceLROnPlateau (when validation metric stalls)

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
  optimizer, mode="max", factor=0.5, patience=2
)
# call scheduler.step(val_metric)
```

---

## Common task recipes

### Multi-class classification: shapes checklist

* Model output: `logits: [B, C]`
* Target: `y: [B]` with dtype `torch.long`
* Loss: `nn.CrossEntropyLoss()` (expects raw logits)

### Binary classification: shapes checklist

* Model output: `logits: [B]` (or `[B, 1]` but be consistent)
* Target: `y: [B]` float in `{0,1}`
* Loss: `nn.BCEWithLogitsLoss()`

### Regression: shapes checklist

* Model output: `pred: [B, 1]`
* Target: `y: [B, 1]` float
* Loss: `MSELoss` or `SmoothL1Loss`

---

## Debugging & iteration playbook (do this under pressure)

1. **Overfit a tiny batch**

* Take 1–4 batches, train for 200–1000 steps.
* If loss won’t drop: bug in data/labels/shapes/loss.

2. **Assert shapes early**

```python
assert x.ndim == 2, x.shape
assert y.ndim == 1, y.shape
assert logits.shape[0] == y.shape[0]
```

3. **Check label range**

```python
# for CE
assert y.min().item() >= 0
assert y.max().item() < logits.size(-1)
```

4. **Check for NaNs**

```python
def has_nan(t):
  return torch.isnan(t).any().item()

assert not has_nan(x)
assert not has_nan(loss)
```

5. **Disable AMP temporarily** if you see instability.

---

## Minimal meters for clean logging

```python
# src/utils/meters.py
class AverageMeter:
  def __init__(self):
    self.sum = 0.0
    self.count = 0

  def update(self, value: float, n: int = 1):
    self.sum += float(value) * n
    self.count += int(n)

  @property
  def avg(self) -> float:
    return self.sum / max(1, self.count)
```

---

## “Nice to have” extras (only if time)

### Exponential Moving Average (EMA) of weights (often boosts validation)

```python
# src/optim/ema.py
import copy
import torch

class EMA:
  def __init__(self, model, decay: float = 0.999):
    self.decay = decay
    self.ema = copy.deepcopy(model).eval()
    for p in self.ema.parameters():
      p.requires_grad_(False)

  @torch.no_grad()
  def update(self, model):
    msd = model.state_dict()
    for k, v in self.ema.state_dict().items():
      if k in msd:
        v.copy_(v * self.decay + msd[k] * (1 - self.decay))
```

### Gradient accumulation (simulate bigger batches)

```python
accum_steps = 4
optimizer.zero_grad(set_to_none=True)

for i, (x, y) in enumerate(loader):
  with torch.autocast(device_type=device.type, enabled=use_amp):
    loss = loss_fn(model(x), y) / accum_steps
  loss.backward()

  if (i + 1) % accum_steps == 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

---

## What to memorize (so you don’t waste time)

* **CE vs BCEWithLogits**: which target dtype/shape each needs.
* **Device moves**: always `.to(device)` for tensors and model.
* **Dataloader speed**: `num_workers`, `pin_memory=True`.
* **Baseline-first**: start with MLP/CNN/Transformer baseline before fancy ideas.
* **One metric = one direction**: decide if “higher is better” and stick to it.

---

## Quick “choose your baseline” mapping

* **Tabular** → `MLP`, `AdamW`, `CrossEntropyLoss` / `MSELoss`
* **Images** → `torchvision.models.resnet18` fine-tune (if allowed) or `SmallCNN`
* **Text / sequences** → `TransformerEncoderModel` + padding mask
* **Time series regression** → Transformer encoder or 1D CNN + `SmoothL1Loss`

---

## Optional: use a pretrained ResNet fast (if allowed)

```python
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def make_resnet18(num_classes: int):
  m = resnet18(weights=ResNet18_Weights.DEFAULT)
  m.fc = nn.Linear(m.fc.in_features, num_classes)
  return m
```

---

## Tiny test: shape sanity (catches dumb bugs fast)

```python
# tests/test_shapes.py
import torch
from src.models.mlp import MLP

def test_mlp_shapes():
  m = MLP(in_dim=10, hidden=[32, 32], out_dim=5)
  x = torch.randn(7, 10)
  y = m(x)
  assert y.shape == (7, 5)
```

---

## Final note

This repo is meant to be a **toolbox**: pick the smallest correct baseline, get a clean training loop running, then iterate on features/modeling choices.

```
```
