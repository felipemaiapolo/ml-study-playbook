# data.py
# Data I/O + validation + splits + PyTorch Dataset/DataLoader helpers.
#
# Design goals:
# - Keep this file focused on "data enters/leaves the system" + batching.
# - Provide simple, reusable utilities you can drop into any interview project.
#
# Dependencies:
# - pandas (recommended)
# - numpy
# - torch
# - pyarrow/fastparquet optional (only needed for parquet/feather I/O)

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union
import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# ---------------------------
# Generic conversions & paths
# ---------------------------

def to_numpy(x: Any) -> np.ndarray:
  """
  Convert torch tensor / list / pandas / numpy to numpy array.
  """
  if hasattr(x, "detach") and callable(x.detach):
    x = x.detach()
  if hasattr(x, "cpu") and callable(x.cpu):
    x = x.cpu()
  if hasattr(x, "numpy") and callable(x.numpy):
    return np.asarray(x.numpy())
  return np.asarray(x)


def ensure_dir(path: Union[str, Path]) -> None:
  Path(path).mkdir(parents=True, exist_ok=True)


def infer_ext(path: Union[str, Path]) -> str:
  return Path(path).suffix.lower()


def cache_path(
  raw_path: Union[str, Path],
  cache_dir: Union[str, Path],
  suffix: str,
  ext: str = ".parquet"
) -> Path:
  """
  Deterministic cache path for derived artifacts:
  cache_path("data/train.csv", "cache", "features_v1") -> cache/train__features_v1.parquet
  """
  raw_path = Path(raw_path)
  cache_dir = Path(cache_dir)
  ensure_dir(cache_dir)
  stem = raw_path.stem
  return cache_dir / f"{stem}__{suffix}{ext}"


# ---------------------------
# JSON / numpy I/O
# ---------------------------

def read_json(path: Union[str, Path]) -> Any:
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def save_json(obj: Any, path: Union[str, Path], indent: int = 2) -> None:
  ensure_dir(Path(path).parent)
  with open(path, "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_npy(path: Union[str, Path]) -> np.ndarray:
  return np.load(path, allow_pickle=False)


def save_npy(arr: Any, path: Union[str, Path]) -> None:
  ensure_dir(Path(path).parent)
  np.save(path, to_numpy(arr), allow_pickle=False)


# ---------------------------
# Table I/O (pandas)
# ---------------------------

def read_table(path: Union[str, Path], **kwargs) -> pd.DataFrame:
  """
  Read a table from common formats based on extension:
  - .csv, .tsv
  - .parquet
  - .feather
  - .pkl/.pickle
  - .json
  """
  path = Path(path)
  ext = path.suffix.lower()

  if ext == ".csv":
    return pd.read_csv(path, **kwargs)
  if ext == ".tsv":
    return pd.read_csv(path, sep="\t", **kwargs)
  if ext == ".parquet":
    try:
      return pd.read_parquet(path, **kwargs)
    except Exception as e:
      raise RuntimeError("Failed to read parquet. Install 'pyarrow' or 'fastparquet'.") from e
  if ext == ".feather":
    try:
      return pd.read_feather(path, **kwargs)
    except Exception as e:
      raise RuntimeError("Failed to read feather. Install 'pyarrow'.") from e
  if ext in (".pkl", ".pickle"):
    return pd.read_pickle(path, **kwargs)
  if ext == ".json":
    return pd.read_json(path, **kwargs)

  raise ValueError(f"Unsupported table format: {ext} ({path})")


def save_table(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
  """
  Save a table to common formats based on extension:
  - .csv, .tsv
  - .parquet
  - .feather
  - .pkl/.pickle
  - .json
  """
  path = Path(path)
  ensure_dir(path.parent)
  ext = path.suffix.lower()

  if ext == ".csv":
    df.to_csv(path, index=False, **kwargs)
    return
  if ext == ".tsv":
    df.to_csv(path, sep="\t", index=False, **kwargs)
    return
  if ext == ".parquet":
    try:
      df.to_parquet(path, index=False, **kwargs)
      return
    except Exception as e:
      raise RuntimeError("Failed to write parquet. Install 'pyarrow' or 'fastparquet'.") from e
  if ext == ".feather":
    try:
      df.to_feather(path, **kwargs)
      return
    except Exception as e:
      raise RuntimeError("Failed to write feather. Install 'pyarrow'.") from e
  if ext in (".pkl", ".pickle"):
    df.to_pickle(path, **kwargs)
    return
  if ext == ".json":
    df.to_json(path, **kwargs)
    return

  raise ValueError(f"Unsupported table format: {ext} ({path})")


# ---------------------------
# Data validation as a first-class step
# ---------------------------

def validate_table(
  df: pd.DataFrame,
  required_cols: Optional[Sequence[str]] = None,
  optional_cols: Optional[Sequence[str]] = None,
  key_cols: Optional[Sequence[str]] = None,
  no_nan_cols: Optional[Sequence[str]] = None,
  time_col: Optional[str] = None,
  require_sorted_time: bool = False,
) -> None:
  """
  One-stop validation you call at the top of every pipeline.

  Typical usage:
    validate_table(
      df,
      required_cols=["date", "y", "store_id"],
      key_cols=["date", "store_id"],
      no_nan_cols=["date", "y"],
      time_col="date",
      require_sorted_time=False
    )

  What it checks (if provided):
  - required columns exist
  - unique key columns (no duplicates)
  - no NaNs in specified columns
  - optionally, time column exists and is sorted
  """
  if required_cols is not None:
    assert_columns(df, required_cols, optional=optional_cols)
  if key_cols is not None:
    check_unique_key(df, key_cols)
  if no_nan_cols is not None:
    check_no_nans(df, no_nan_cols)
  if time_col is not None:
    if time_col not in df.columns:
      raise ValueError(f"time_col='{time_col}' not in df.columns")
    if require_sorted_time:
      check_sorted_by_time(df, time_col)


def assert_columns(
  df: pd.DataFrame,
  required: Sequence[str],
  optional: Optional[Sequence[str]] = None
) -> None:
  missing = [c for c in required if c not in df.columns]
  if missing:
    raise ValueError(f"Missing required columns: {missing}")
  # optional is intentionally ignored if missing


def check_no_nans(df: pd.DataFrame, cols: Sequence[str]) -> None:
  bad = [c for c in cols if df[c].isna().any()]
  if bad:
    raise ValueError(f"NaNs detected in columns: {bad}")


def check_unique_key(df: pd.DataFrame, key_cols: Sequence[str]) -> None:
  if df.duplicated(list(key_cols)).any():
    dup = df[df.duplicated(list(key_cols), keep=False)].head(5)
    raise ValueError(
      f"Duplicate keys found for key_cols={list(key_cols)}. Example rows:\n{dup}"
    )


def check_sorted_by_time(df: pd.DataFrame, time_col: str) -> None:
  if not df[time_col].is_monotonic_increasing:
    raise ValueError(f"Data is not sorted by {time_col} (monotonic increasing).")


# ---------------------------
# Split helpers
# ---------------------------

def time_split(
  df: pd.DataFrame,
  time_col: str,
  val_size: Optional[int] = None,
  test_size: Optional[int] = None,
  val_start: Optional[Any] = None,
  test_start: Optional[Any] = None,
  sort: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """
  Split a dataframe by time into (train, val, test).

  Specify either sizes (val_size/test_size) or explicit start timestamps (val_start/test_start).
  """
  if sort:
    df = df.sort_values(time_col).reset_index(drop=True)

  if val_start is not None or test_start is not None:
    if val_start is None:
      raise ValueError("If using start times, val_start must be provided.")
    train = df[df[time_col] < val_start]
    rest = df[df[time_col] >= val_start]
    if test_start is None:
      val = rest
      test = df.iloc[0:0]
    else:
      val = rest[rest[time_col] < test_start]
      test = rest[rest[time_col] >= test_start]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

  if val_size is None:
    raise ValueError("Provide val_size/test_size or val_start/test_start.")

  n = len(df)
  test_size = test_size or 0
  if val_size + test_size >= n:
    raise ValueError("val_size + test_size must be < n")

  train_end = n - (val_size + test_size)
  val_end = n - test_size

  train = df.iloc[:train_end]
  val = df.iloc[train_end:val_end]
  test = df.iloc[val_end:]

  return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def rolling_time_splits(
  df: pd.DataFrame,
  time_col: str,
  train_window: int,
  val_window: int,
  step: int,
  sort: bool = True
):
  """
  Yield rolling (train, val) splits by time on sorted rows.
  """
  if sort:
    df = df.sort_values(time_col).reset_index(drop=True)

  n = len(df)
  start = 0
  while True:
    train_start = start
    train_end = train_start + train_window
    val_end = train_end + val_window
    if val_end > n:
      break
    train = df.iloc[train_start:train_end]
    val = df.iloc[train_end:val_end]
    yield train.reset_index(drop=True), val.reset_index(drop=True)
    start += step


def group_split(
  df: pd.DataFrame,
  group_col: str,
  val_frac: float = 0.2,
  seed: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """
  Split by group ids to reduce leakage across entities.
  """
  if not (0.0 < val_frac < 1.0):
    raise ValueError("val_frac must be in (0,1)")

  groups = df[group_col].dropna().unique().tolist()
  rng = np.random.default_rng(seed)
  rng.shuffle(groups)

  n_val = max(1, int(round(val_frac * len(groups))))
  val_groups = set(groups[:n_val])

  val = df[df[group_col].isin(val_groups)]
  train = df[~df[group_col].isin(val_groups)]

  return train.reset_index(drop=True), val.reset_index(drop=True)

def random_split_xy(
  X,
  y,
  val_frac: float = 0.2,
  test_frac: float = 0.1,
  seed: int = 0,
  stratify: bool = False
):
  """
  Randomly split (X, y) into train/val/test using sklearn's train_test_split.

  Inputs:
    X: array-like, shape (n, d_x) (or any indexable array-like)
    y: array-like, shape (n, d_y) or (n,) or (n,1) or one-hot (n,C)
    val_frac: fraction of full data to use for validation
    test_frac: fraction of full data to use for test
    seed: random_state
    stratify: if True, stratify splits using class labels inferred from y
      - intended for single-label classification

  Returns:
    X_train, y_train, X_val, y_val, X_test, y_test (all numpy arrays)
  """
  Xn = np.asarray(X)
  yn = np.asarray(y)

  if Xn.shape[0] != yn.shape[0]:
    raise ValueError(f"Mismatched n: X has {Xn.shape[0]}, y has {yn.shape[0]}")

  if not (0.0 < val_frac < 1.0):
    raise ValueError("val_frac must be in (0,1)")
  if not (0.0 <= test_frac < 1.0):
    raise ValueError("test_frac must be in [0,1)")
  if val_frac + test_frac >= 1.0:
    raise ValueError("val_frac + test_frac must be < 1")

  strat_labels = None
  if stratify:
    if yn.ndim == 1:
      strat_labels = yn.astype(int)
    elif yn.ndim == 2 and yn.shape[1] == 1:
      strat_labels = yn.reshape(-1).astype(int)
    else:
      raise ValueError(f"Cannot infer stratification labels from y with shape {yn.shape}")

  # 1) Split out test
  X_tmp, X_test, y_tmp, y_test = train_test_split(
    Xn,
    yn,
    test_size=test_frac,
    random_state=seed,
    shuffle=True,
    stratify=strat_labels
  )

  # 2) Split tmp into train/val (val_frac relative to tmp)
  val_frac_tmp = val_frac / (1.0 - test_frac)
  if stratify:
    # Need labels for tmp, consistent with y_tmp
    y_tmp_arr = np.asarray(y_tmp)
    if y_tmp_arr.ndim == 1:
      strat_tmp = y_tmp_arr.astype(int)
    elif y_tmp_arr.ndim == 2 and y_tmp_arr.shape[1] == 1:
      strat_tmp = y_tmp_arr.reshape(-1).astype(int)
    else:
      strat_tmp = np.argmax(y_tmp_arr, axis=1).astype(int)
  else:
    strat_tmp = None

  X_train, X_val, y_train, y_val = train_test_split(
    X_tmp,
    y_tmp,
    test_size=val_frac_tmp,
    random_state=seed,
    shuffle=True,
    stratify=strat_tmp
  )

  return (
    np.asarray(X_train), np.asarray(y_train),
    np.asarray(X_val), np.asarray(y_val),
    np.asarray(X_test), np.asarray(y_test)
  )

# ---------------------------
# DataFrame -> numpy helpers
# ---------------------------

def df_to_xy(
  df: pd.DataFrame,
  feature_cols: Sequence[str],
  target_cols: Sequence[str],
  dtype_x: Any = np.float32,
  dtype_y: Any = np.float32
) -> Tuple[np.ndarray, np.ndarray]:
  """
  Convert a dataframe into (X, y) numpy arrays.
  Targets are always returned as 2D (n, d_y).
  """
  X = df.loc[:, list(feature_cols)].to_numpy(dtype=dtype_x, copy=True)
  y = df.loc[:, list(target_cols)].to_numpy(dtype=dtype_y, copy=True)
  if y.ndim == 1:
    y = y.reshape(-1, 1)
  return X, y


# ---------------------------
# Torch datasets
# ---------------------------

class TabularDataset(Dataset):
  """
  Generic tabular dataset for torch.

  - X: (n, d_x)
  - y: (n, d_y) for regression, or (n, 1)/(n,C) for classification
  - transform_x/transform_y are applied after tensor conversion.
  """
  def __init__(
    self,
    X: Any,
    y: Any,
    task: str = "regression",  # "regression" or "classification"
    transform_x: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    transform_y: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
  ):
    if task not in ("regression", "classification"):
      raise ValueError("task must be 'regression' or 'classification'")

    Xn = to_numpy(X)
    yn = to_numpy(y)

    if Xn.ndim != 2:
      raise ValueError(f"X must be 2D (n,d). Got {Xn.shape}")
    if yn.ndim == 1:
      yn = yn.reshape(-1, 1)
    if yn.ndim != 2:
      raise ValueError(f"y must be 2D (n,dy). Got {yn.shape}")
    if Xn.shape[0] != yn.shape[0]:
      raise ValueError(f"Mismatched n: X has {Xn.shape[0]}, y has {yn.shape[0]}")

    self.task = task
    self.transform_x = transform_x
    self.transform_y = transform_y

    self.X = torch.tensor(Xn, dtype=torch.float32)

    if task == "regression":
      self.y = torch.tensor(yn, dtype=torch.float32)
    else:
      if yn.shape[1] == 1:
        labels = yn.reshape(-1).astype(np.int64)
      else:
        labels = np.argmax(yn, axis=1).astype(np.int64)
      self.y = torch.tensor(labels, dtype=torch.long)

  def __len__(self) -> int:
    return self.X.shape[0]

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x = self.X[idx]
    y = self.y[idx]
    if self.transform_x is not None:
      x = self.transform_x(x)
    if self.transform_y is not None:
      y = self.transform_y(y)
    return x, y


class TextDataset(Dataset):
  """
  Dataset for text classification/regression.

  - texts: list-like length n
  - y: (n, dy) for regression, or (n,1)/(n,C) for classification
  - tokenize: optional callable(text) -> dict/tensor
    If provided, __getitem__ returns (tokenized, y) where tokenized can be a dict.
  """
  def __init__(
    self,
    texts: Sequence[str],
    y: Any,
    task: str = "classification",
    tokenize: Optional[Callable[[str], Any]] = None
  ):
    if task not in ("regression", "classification"):
      raise ValueError("task must be 'regression' or 'classification'")
    self.texts = list(texts)
    self.task = task
    self.tokenize = tokenize

    yn = to_numpy(y)
    if yn.ndim == 1:
      yn = yn.reshape(-1, 1)
    if len(self.texts) != yn.shape[0]:
      raise ValueError(f"Mismatched n: texts {len(self.texts)} vs y {yn.shape[0]}")

    if task == "regression":
      self.y = torch.tensor(yn, dtype=torch.float32)
    else:
      if yn.shape[1] == 1:
        labels = yn.reshape(-1).astype(np.int64)
      else:
        labels = np.argmax(yn, axis=1).astype(np.int64)
      self.y = torch.tensor(labels, dtype=torch.long)

  def __len__(self) -> int:
    return len(self.texts)

  def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor]:
    text = self.texts[idx]
    y = self.y[idx]
    if self.tokenize is None:
      return text, y
    return self.tokenize(text), y


class SequenceDataset(Dataset):
  """
  Sliding-window sequence dataset for time series.

  Output per item:
    X_window: (L, d_x)
    y_target: (d_y,)
  """
  def __init__(
    self,
    X: Any,
    y: Optional[Any] = None,
    lookback: int = 30,
    horizon: int = 1,
    stride: int = 1,
    dtype_x: torch.dtype = torch.float32,
    dtype_y: torch.dtype = torch.float32
  ):
    Xn = to_numpy(X)
    if Xn.ndim != 2:
      raise ValueError(f"X must be 2D (n,dx). Got {Xn.shape}")

    if y is None:
      yn = Xn
    else:
      yn = to_numpy(y)
      if yn.ndim == 1:
        yn = yn.reshape(-1, 1)
      if yn.ndim != 2:
        raise ValueError(f"y must be 2D (n,dy). Got {yn.shape}")
      if yn.shape[0] != Xn.shape[0]:
        raise ValueError("X and y must have same n")

    if lookback <= 0 or horizon <= 0 or stride <= 0:
      raise ValueError("lookback, horizon, stride must be positive")

    self.X = torch.tensor(Xn, dtype=dtype_x)
    self.y = torch.tensor(yn, dtype=dtype_y)
    self.lookback = lookback
    self.horizon = horizon
    self.stride = stride

    self._t_min = lookback
    self._t_max = Xn.shape[0] - horizon + 1
    if self._t_max <= self._t_min:
      raise ValueError("Not enough data for given lookback/horizon")

    self._indices = list(range(self._t_min, self._t_max, stride))

  def __len__(self) -> int:
    return len(self._indices)

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    t = self._indices[idx]
    xw = self.X[t - self.lookback:t]
    yt = self.y[t + self.horizon - 1]
    return xw, yt


# ---------------------------
# Collate functions
# ---------------------------

def collate_text_raw(batch: List[Tuple[str, torch.Tensor]]) -> Tuple[List[str], torch.Tensor]:
  texts = [b[0] for b in batch]
  ys = torch.stack([b[1] for b in batch], dim=0)
  return texts, ys


def collate_tokenized_dict(batch: List[Tuple[Dict[str, Any], torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
  xs = [b[0] for b in batch]
  ys = torch.stack([b[1] for b in batch], dim=0)

  keys = xs[0].keys()
  out: Dict[str, torch.Tensor] = {}
  for k in keys:
    vals = [x[k] for x in xs]
    vals_t = [v if torch.is_tensor(v) else torch.tensor(v) for v in vals]
    out[k] = torch.stack(vals_t, dim=0)

  return out, ys


# ---------------------------
# DataLoader reproducibility
# ---------------------------

def seed_worker(worker_id: int) -> None:
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)


def make_dataloader(
  dataset: Dataset,
  batch_size: int,
  shuffle: bool = True,
  num_workers: int = 0,
  drop_last: bool = False,
  pin_memory: Optional[bool] = None,
  persistent_workers: Optional[bool] = None,
  collate_fn: Optional[Callable] = None,
  seed: Optional[int] = None
) -> DataLoader:
  if pin_memory is None:
    pin_memory = torch.cuda.is_available()
  if persistent_workers is None:
    persistent_workers = num_workers > 0

  gen = None
  if seed is not None:
    gen = torch.Generator()
    gen.manual_seed(int(seed))

  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    drop_last=drop_last,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
    collate_fn=collate_fn,
    worker_init_fn=seed_worker if num_workers > 0 else None,
    generator=gen,
  )


def make_loaders(
  train_ds: Dataset,
  val_ds: Optional[Dataset],
  batch_size: int,
  num_workers: int = 0,
  seed: Optional[int] = None,
  collate_fn: Optional[Callable] = None
) -> Tuple[DataLoader, Optional[DataLoader]]:
  train_loader = make_dataloader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=collate_fn,
    seed=seed
  )
  val_loader = None
  if val_ds is not None:
    val_loader = make_dataloader(
      val_ds,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      collate_fn=collate_fn,
      seed=seed
    )
  return train_loader, val_loader

