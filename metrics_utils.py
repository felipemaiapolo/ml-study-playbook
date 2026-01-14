# metrics.py
# Scikit-learn-first metrics helpers + SciPy for correlations (Pearson/Spearman) + calibration.
#
# Conventions:
# - Inputs can be numpy / list / torch tensors (auto-converted to numpy).
# - y_score / y_hat and y are expected to be 2D arrays; 1D will be reshaped.
# - Three public functions:
#   1) binary_classification_metrics
#   2) multiclass_classification_metrics
#   3) regression_metrics
#
# Calibration:
# - Binary: Brier score + calibration curve (10 quantile bins) + average |gap| to diagonal.
# - Multiclass: multiclass Brier + per-class OVR calibration curves/gaps + aggregated (macro/weighted/micro) gaps.

from typing import Any, Dict, List, Tuple, Optional
import numpy as np

from sklearn.metrics import (
  accuracy_score,
  precision_recall_fscore_support,
  log_loss,
  roc_auc_score,
  mean_squared_error,
  mean_absolute_error,
  brier_score_loss,
)

from sklearn.calibration import calibration_curve
from scipy.stats import spearmanr, pearsonr
# Add these imports near the top of metrics.py (if not already present)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

_EPS = 1e-12

def _to_numpy(x: Any) -> np.ndarray:
  # Torch -> numpy (without importing torch)
  if hasattr(x, "detach") and callable(x.detach):
    x = x.detach()
  if hasattr(x, "cpu") and callable(x.cpu):
    x = x.cpu()
  if hasattr(x, "numpy") and callable(x.numpy):
    return np.asarray(x.numpy())
  return np.asarray(x)


def _ensure_2d(a: np.ndarray) -> np.ndarray:
  a = np.asarray(a)
  if a.ndim == 1:
    return a.reshape(-1, 1)
  if a.ndim != 2:
    raise ValueError(f"Expected 2D array (or 1D), got shape {a.shape}")
  return a


def _softmax(logits: np.ndarray, axis: int = 1) -> np.ndarray:
  logits = np.asarray(logits, dtype=np.float64)
  m = np.max(logits, axis=axis, keepdims=True)
  exps = np.exp(logits - m)
  return exps / np.clip(np.sum(exps, axis=axis, keepdims=True), _EPS, None)


def _sigmoid(z: np.ndarray) -> np.ndarray:
  z = np.asarray(z, dtype=np.float64)
  out = np.empty_like(z, dtype=np.float64)
  pos = z >= 0
  neg = ~pos
  out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
  ez = np.exp(z[neg])
  out[neg] = ez / (1.0 + ez)
  return out


def _labels_from_y(y: np.ndarray) -> np.ndarray:
  y = _ensure_2d(y)
  if y.shape[1] == 1:
    return y.reshape(-1).astype(int)
  return np.argmax(y, axis=1).astype(int)


def _onehot_from_labels(labels: np.ndarray, n_classes: int) -> np.ndarray:
  labels = labels.astype(int)
  oh = np.zeros((labels.shape[0], n_classes), dtype=int)
  oh[np.arange(labels.shape[0]), labels] = 1
  return oh


def _pearson_1d(a: np.ndarray, b: np.ndarray) -> float:
  a = np.asarray(a, dtype=np.float64).reshape(-1)
  b = np.asarray(b, dtype=np.float64).reshape(-1)
  try:
    r, _ = pearsonr(a, b)
    return float(r)
  except Exception:
    return float("nan")


def _spearman_1d(a: np.ndarray, b: np.ndarray) -> float:
  a = np.asarray(a, dtype=np.float64).reshape(-1)
  b = np.asarray(b, dtype=np.float64).reshape(-1)
  try:
    res = spearmanr(a, b)
    corr = getattr(res, "correlation", res)
    return float(corr)
  except Exception:
    return float("nan")


def _calibration_stats_binary(
  y_true01: np.ndarray,
  p_pos: np.ndarray,
  n_bins: int = 10
) -> Tuple[float, List[float], List[float]]:
  """
  Returns:
    - avg_abs_gap: mean(|prob_true - prob_pred|) across non-empty bins
    - prob_pred_list
    - prob_true_list
  """
  y_true01 = np.asarray(y_true01).astype(int).reshape(-1)
  p_pos = np.asarray(p_pos, dtype=np.float64).reshape(-1)
  p_pos = np.clip(p_pos, _EPS, 1.0 - _EPS)

  # calibration_curve can fail if only one class present
  prob_true, prob_pred = calibration_curve(
    y_true01,
    p_pos,
    n_bins=n_bins,
    strategy="quantile"
  )
  gap = np.abs(prob_true - prob_pred)
  avg_abs_gap = float(np.mean(gap)) if gap.size > 0 else float("nan")
  return avg_abs_gap, [float(x) for x in prob_pred], [float(x) for x in prob_true]


def binary_classification_metrics(
  y_score: Any,
  y: Any,
  threshold: float = 0.5,
  input_type: str = "probs",  # "probs" or "logits"
  pos_index: int = 1,
  n_calib_bins: int = 10
) -> Dict[str, float]:
  """
  Binary classification metrics.

  Inputs:
    y_score: (n,1) prob/logit for positive class OR (n,2) probs/logits for both classes.
    y: (n,1) labels in {0,1} OR (n,2) one-hot.
  Returns:
    dict with:
      acc/precision/recall/f1 at threshold,
      cross_entropy, auc,
      brier,
      calib_abs_gap (quantile bins),
      calib_curve_prob_pred, calib_curve_prob_true (lists)
  """
  ys = _ensure_2d(_to_numpy(y_score))
  yy = _ensure_2d(_to_numpy(y))
  y_true = _labels_from_y(yy)

  if input_type not in ("probs", "logits"):
    raise ValueError("input_type must be 'probs' or 'logits'")

  if ys.shape[1] == 1:
    if input_type == "probs":
      p_pos = ys.reshape(-1).astype(np.float64)
    else:
      p_pos = _sigmoid(ys.reshape(-1))
  elif ys.shape[1] == 2:
    if pos_index not in (0, 1):
      raise ValueError("pos_index must be 0 or 1 for binary (n,2) scores.")
    if input_type == "probs":
      probs2 = ys.astype(np.float64)
    else:
      probs2 = _softmax(ys, axis=1)
    p_pos = probs2[:, pos_index]
  else:
    raise ValueError(f"Binary expects y_score with 1 or 2 columns, got {ys.shape[1]}")

  p_pos = np.clip(p_pos, _EPS, 1.0 - _EPS)
  y_pred = (p_pos >= threshold).astype(int)

  tp = int(np.sum((y_true == 1) & (y_pred == 1)))
  fp = int(np.sum((y_true == 0) & (y_pred == 1)))
  fn = int(np.sum((y_true == 1) & (y_pred == 0)))
  tn = int(np.sum((y_true == 0) & (y_pred == 0)))

  precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
  recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
  f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
  acc = (tp + tn) / max(1, (tp + tn + fp + fn))

  out: Dict[str, float] = {}
  out[f"acc@{threshold:g}"] = float(acc)
  out[f"precision@{threshold:g}"] = float(precision)
  out[f"recall@{threshold:g}"] = float(recall)
  out[f"f1@{threshold:g}"] = float(f1)

  # threshold-free metrics
  try:
    out["cross_entropy"] = float(log_loss(y_true, p_pos, labels=[0, 1]))
  except Exception:
    out["cross_entropy"] = float("nan")

  try:
    out["auc"] = float(roc_auc_score(y_true, p_pos))
  except Exception:
    out["auc"] = float("nan")

  # calibration: Brier + calibration curve gap
  try:
    out["brier"] = float(brier_score_loss(y_true, p_pos))
  except Exception:
    out["brier"] = float("nan")

  try:
    calib_gap, prob_pred, prob_true = _calibration_stats_binary(y_true, p_pos, n_bins=n_calib_bins)
    out["calib_abs_gap"] = float(calib_gap)
    # "calibration curves" as points: (prob_pred, prob_true)
    out["calib_curve_prob_pred"] = prob_pred
    out["calib_curve_prob_true"] = prob_true
  except Exception:
    out["calib_abs_gap"] = float("nan")
    out["calib_curve_prob_pred"] = []
    out["calib_curve_prob_true"] = []

  return out


def multiclass_classification_metrics(
  y_score: Any,
  y: Any,
  input_type: str = "probs",     # "probs" or "logits"
  multi_class_auc: str = "ovr",  # "ovr" or "ovo"
  n_calib_bins: int = 10
) -> Dict[str, float]:
  """
  Multinomial (single-label) classification metrics.

  Inputs:
    y_score: (n,C) probs or logits
    y: (n,1) integer labels OR (n,C) one-hot
  Returns:
    dict with:
      acc,
      precision/recall/f1 (macro/micro/weighted),
      cross_entropy,
      auc_(macro/micro/weighted),
      brier (multiclass),
      calib_abs_gap_(macro/micro/weighted) computed via OVR calibration curves,
      plus calibration curve points (per-class + micro).
  """
  ys = _ensure_2d(_to_numpy(y_score))
  yy = _ensure_2d(_to_numpy(y))
  y_true = _labels_from_y(yy)

  if ys.shape[1] < 2:
    raise ValueError(f"Multiclass expects y_score with C>=2 columns, got {ys.shape[1]}")

  if input_type not in ("probs", "logits"):
    raise ValueError("input_type must be 'probs' or 'logits'")

  if multi_class_auc not in ("ovr", "ovo"):
    raise ValueError("multi_class_auc must be 'ovr' or 'ovo'")

  if input_type == "probs":
    probs = ys.astype(np.float64)
  else:
    probs = _softmax(ys, axis=1)

  probs = np.clip(probs, _EPS, 1.0)
  probs = probs / np.clip(probs.sum(axis=1, keepdims=True), _EPS, None)

  n_classes = probs.shape[1]
  y_pred = np.argmax(probs, axis=1)
  y_onehot = _onehot_from_labels(y_true, n_classes)

  out: Dict[str, float] = {}
  out["acc"] = float(accuracy_score(y_true, y_pred))

  for avg in ("macro", "micro", "weighted"):
    p, r, f, _ = precision_recall_fscore_support(
      y_true,
      y_pred,
      average=avg,
      zero_division=0
    )
    out[f"precision_{avg}"] = float(p)
    out[f"recall_{avg}"] = float(r)
    out[f"f1_{avg}"] = float(f)

  # Cross-entropy
  try:
    out["cross_entropy"] = float(log_loss(y_true, probs, labels=list(range(n_classes))))
  except Exception:
    out["cross_entropy"] = float("nan")

  # AUC macro/weighted (sklearn multiclass)
  for avg in ("macro", "weighted"):
    try:
      out[f"auc_{avg}"] = float(
        roc_auc_score(y_true, probs, multi_class=multi_class_auc, average=avg)
      )
    except Exception:
      out[f"auc_{avg}"] = float("nan")

  # AUC micro via flattening (robust across sklearn versions)
  try:
    out["auc_micro"] = float(roc_auc_score(y_onehot.ravel(), probs.ravel()))
  except Exception:
    out["auc_micro"] = float("nan")

  # Multiclass Brier score: mean over samples of sum_k (p_k - y_k)^2
  try:
    out["brier"] = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))
  except Exception:
    out["brier"] = float("nan")

  # Calibration (OVR per class + aggregated gaps)
  gaps_by_class: List[float] = []
  curve_pred_by_class: List[List[float]] = []
  curve_true_by_class: List[List[float]] = []

  class_counts = np.bincount(y_true, minlength=n_classes).astype(np.float64)
  class_weights = class_counts / max(1.0, float(np.sum(class_counts)))

  for k in range(n_classes):
    yk = y_onehot[:, k]
    pk = probs[:, k]
    try:
      gap_k, prob_pred, prob_true = _calibration_stats_binary(yk, pk, n_bins=n_calib_bins)
      gaps_by_class.append(float(gap_k))
      curve_pred_by_class.append(prob_pred)
      curve_true_by_class.append(prob_true)
    except Exception:
      gaps_by_class.append(float("nan"))
      curve_pred_by_class.append([])
      curve_true_by_class.append([])

  out["calib_abs_gap_by_class"] = [float(x) if np.isfinite(x) else float("nan") for x in gaps_by_class]
  out["calib_curve_prob_pred_by_class"] = curve_pred_by_class
  out["calib_curve_prob_true_by_class"] = curve_true_by_class

  # macro / weighted gaps (ignore NaNs)
  gaps_arr = np.asarray(gaps_by_class, dtype=np.float64)
  out["calib_abs_gap_macro"] = float(np.nanmean(gaps_arr)) if np.isfinite(gaps_arr).any() else float("nan")

  try:
    # Weighted average ignoring NaNs by re-normalizing weights over finite gaps
    finite = np.isfinite(gaps_arr)
    if not finite.any():
      out["calib_abs_gap_weighted"] = float("nan")
    else:
      w = class_weights[finite]
      w = w / np.clip(np.sum(w), _EPS, None)
      out["calib_abs_gap_weighted"] = float(np.sum(w * gaps_arr[finite]))
  except Exception:
    out["calib_abs_gap_weighted"] = float("nan")

  # micro calibration: flatten one-hot and probs, then do binary calibration
  try:
    gap_micro, prob_pred_micro, prob_true_micro = _calibration_stats_binary(
      y_onehot.ravel(),
      probs.ravel(),
      n_bins=n_calib_bins
    )
    out["calib_abs_gap_micro"] = float(gap_micro)
    out["calib_curve_prob_pred_micro"] = prob_pred_micro
    out["calib_curve_prob_true_micro"] = prob_true_micro
  except Exception:
    out["calib_abs_gap_micro"] = float("nan")
    out["calib_curve_prob_pred_micro"] = []
    out["calib_curve_prob_true_micro"] = []

  return out


def regression_metrics(
  y_hat: Any,
  y: Any
) -> Dict[str, float]:
  """
  Regression metrics with support for multi-output regression.

  Inputs:
    y_hat: (n,d) predictions (or (n,1))
    y: (n,d) targets (or (n,1))
  Returns:
    dict with overall means + per-dimension arrays:
      mse, rmse, mae, pearson, spearman
  """
  yp = _ensure_2d(_to_numpy(y_hat)).astype(np.float64)
  yt = _ensure_2d(_to_numpy(y)).astype(np.float64)

  if yp.shape != yt.shape:
    raise ValueError(f"y_hat and y must have same shape. Got {yp.shape} vs {yt.shape}")

  mse_by = mean_squared_error(yt, yp, multioutput="raw_values")
  mae_by = mean_absolute_error(yt, yp, multioutput="raw_values")
  rmse_by = np.sqrt(np.maximum(mse_by, 0.0))

  pearson_by = []
  spearman_by = []
  for j in range(yt.shape[1]):
    pearson_by.append(_pearson_1d(yt[:, j], yp[:, j]))
    spearman_by.append(_spearman_1d(yt[:, j], yp[:, j]))

  mse_by = np.asarray(mse_by, dtype=np.float64)
  rmse_by = np.asarray(rmse_by, dtype=np.float64)
  mae_by = np.asarray(mae_by, dtype=np.float64)
  pearson_by = np.asarray(pearson_by, dtype=np.float64)
  spearman_by = np.asarray(spearman_by, dtype=np.float64)

  out: Dict[str, float] = {}
  out["mse"] = float(np.nanmean(mse_by))
  out["rmse"] = float(np.nanmean(rmse_by))
  out["mae"] = float(np.nanmean(mae_by))
  out["pearson"] = float(np.nanmean(pearson_by))
  out["spearman"] = float(np.nanmean(spearman_by))

  out["mse_by_dim"] = [float(x) for x in mse_by]
  out["rmse_by_dim"] = [float(x) for x in rmse_by]
  out["mae_by_dim"] = [float(x) for x in mae_by]
  out["pearson_by_dim"] = [float(x) if np.isfinite(x) else float("nan") for x in pearson_by]
  out["spearman_by_dim"] = [float(x) if np.isfinite(x) else float("nan") for x in spearman_by]

  return out

## Outputsa
def plot_confusion_matrix(
  y: Any,
  y_score: Any = None,
  y_pred: Any = None,
  task: str = "multiclass",          # "binary" or "multiclass"
  input_type: str = "probs",         # "probs" or "logits" (only used if y_score is given)
  threshold: float = 0.5,            # only used for binary if y_score is given
  pos_index: int = 1,                # only used for binary when y_score has shape (n,2)
  normalize: Optional[str] = None,   # None, "true", "pred", "all"
  labels: Optional[List[int]] = None,
  display_labels: Optional[List[str]] = None,
  ax: Optional[Any] = None,
  title: Optional[str] = None
) -> Tuple[np.ndarray, Any, Any]:
  """
  Plot and return confusion matrix.

  You can provide either:
    - y_pred (preferred if already computed), or
    - y_score (logits/probs) to derive y_pred

  Inputs:
    y: (n,1) labels or (n,C) one-hot
    y_score: (n,1)/(n,2) for binary or (n,C) for multiclass
    y_pred: (n,1) predicted labels (or (n,) is fine)
  Returns:
    cm: np.ndarray
    fig, ax
  """
  yy = _ensure_2d(_to_numpy(y))
  y_true = _labels_from_y(yy)

  if y_pred is None:
    if y_score is None:
      raise ValueError("Provide either y_pred or y_score.")
    ys = _ensure_2d(_to_numpy(y_score))

    if task == "binary":
      if input_type not in ("probs", "logits"):
        raise ValueError("input_type must be 'probs' or 'logits'")

      if ys.shape[1] == 1:
        p_pos = ys.reshape(-1).astype(np.float64)
        if input_type == "logits":
          p_pos = _sigmoid(p_pos)
      elif ys.shape[1] == 2:
        if pos_index not in (0, 1):
          raise ValueError("pos_index must be 0 or 1 for binary (n,2) scores.")
        probs2 = ys.astype(np.float64) if input_type == "probs" else _softmax(ys, axis=1)
        p_pos = probs2[:, pos_index]
      else:
        raise ValueError(f"Binary expects y_score with 1 or 2 columns, got {ys.shape[1]}")

      y_pred_lbl = (p_pos >= threshold).astype(int)

    elif task == "multiclass":
      if ys.shape[1] < 2:
        raise ValueError(f"Multiclass expects y_score with C>=2 columns, got {ys.shape[1]}")
      probs = ys.astype(np.float64) if input_type == "probs" else _softmax(ys, axis=1)
      y_pred_lbl = np.argmax(probs, axis=1).astype(int)

    else:
      raise ValueError("task must be 'binary' or 'multiclass'")
  else:
    yp = _to_numpy(y_pred)
    if yp.ndim == 2 and yp.shape[1] == 1:
      yp = yp.reshape(-1)
    y_pred_lbl = yp.astype(int)

  cm = confusion_matrix(
    y_true,
    y_pred_lbl,
    labels=labels,
    normalize=normalize
  )

  if ax is None:
    fig, ax = plt.subplots()
  else:
    fig = ax.figure

  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
  disp.plot(ax=ax, values_format=".2f" if normalize else "d", colorbar=False)

  if title is None:
    base = "Confusion Matrix"
    if normalize:
      base += f" (normalize={normalize})"
    title = base
  ax.set_title(title)
  fig.tight_layout()

  return cm, fig, ax


def compare_models(
  y_hats: List[Any],
  names: List[str],
  y: Any,
  task: str,                      # "binary", "multiclass", "regression"
  input_type: str = "probs",       # for classification: "probs" or "logits"
  threshold: float = 0.5,          # for binary threshold metrics
  pos_index: int = 1,              # binary (n,2) case
  multi_class_auc: str = "ovr",    # multiclass AUC mode
  n_calib_bins: int = 10,
  metrics: Optional[List[str]] = None,
  sort_by: Optional[str] = None,
  ascending: bool = False,
  print_table: bool = True
) -> pd.DataFrame:
  """
  Compare multiple models by computing metrics for each and printing a table.

  Inputs:
    y_hats: list of predictions (each can be torch/numpy). Shapes:
      - binary: (n,1) or (n,2) probs/logits
      - multiclass: (n,C) probs/logits
      - regression: (n,d)
    names: list of model names (same length as y_hats)
    y: ground-truth (2D, but 1D tolerated)
    metrics: optional list of column names to keep in the output table
    sort_by: optional metric column to sort by
  Returns:
    pandas DataFrame
  """
  if len(y_hats) != len(names):
    raise ValueError("y_hats and names must have same length")

  rows = []
  for yh, name in zip(y_hats, names):
    if task == "binary":
      d = binary_classification_metrics(
        y_score=yh,
        y=y,
        threshold=threshold,
        input_type=input_type,
        pos_index=pos_index,
        n_calib_bins=n_calib_bins
      )
      # Keep a sane default subset if caller didn't specify
      default_cols = [
        f"acc@{threshold:g}",
        f"f1@{threshold:g}",
        f"precision@{threshold:g}",
        f"recall@{threshold:g}",
        "auc",
        "cross_entropy",
        "brier",
        "calib_abs_gap",
      ]
    elif task == "multiclass":
      d = multiclass_classification_metrics(
        y_score=yh,
        y=y,
        input_type=input_type,
        multi_class_auc=multi_class_auc,
        n_calib_bins=n_calib_bins
      )
      default_cols = [
        "acc",
        "f1_macro",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
        "auc_macro",
        "auc_weighted",
        "auc_micro",
        "cross_entropy",
        "brier",
        "calib_abs_gap_macro",
        "calib_abs_gap_weighted",
        "calib_abs_gap_micro",
      ]
    elif task == "regression":
      d = regression_metrics(y_hat=yh, y=y)
      default_cols = [
        "rmse",
        "mae",
        "mse",
        "pearson",
        "spearman",
      ]
    else:
      raise ValueError("task must be 'binary', 'multiclass', or 'regression'")

    row = {"model": name}
    row.update(d)
    rows.append(row)

  df = pd.DataFrame(rows)

  # If user didn't pass a metric list, keep defaults (plus model)
  if metrics is None:
    if len(rows) > 0:
      if task in ("binary", "multiclass"):
        keep = ["model"] + [c for c in default_cols if c in df.columns]
      else:
        keep = ["model"] + [c for c in default_cols if c in df.columns]
      df = df.loc[:, keep]
  else:
    keep = ["model"] + [m for m in metrics if m in df.columns]
    df = df.loc[:, keep]

  if sort_by is not None and sort_by in df.columns:
    df = df.sort_values(sort_by, ascending=bool(ascending)).reset_index(drop=True)
  else:
    df = df.reset_index(drop=True)

  if print_table:
    # Print nicely without scientific spam
    with pd.option_context("display.max_columns", None, "display.width", 200):
      print(df.to_string(index=False))

  return df

# -----------------------
# Suggestions (keep these in mind when wiring metrics into training)
# -----------------------
#
# 1) Always log what you fed into metrics:
#    - input_type ('logits' vs 'probs'), threshold, pos_index, and C (#classes).
#    Most "weird metrics" are just mismatched conventions.
#
# 2) AUC and calibration can be undefined on small/biased splits:
#    - If your validation set has only one class (binary) or misses classes (multiclass),
#      roc_auc_score / calibration_curve may error and you'll get NaNs. That's normal.
#      Don't "fix" it by peeking; fix your split or report the limitation.
#
# 3) Cross-entropy reporting vs training:
#    - During training, compute CE in torch (CrossEntropyLoss/BCEWithLogitsLoss) for stability.
#      Use these functions for evaluation/reporting (post-hoc).
#
# 4) Calibration:
#    - The single-number 'calib_abs_gap' is mean(|empirical_pos_rate - mean_pred_prob|) across bins.
#      It’s a simple sanity check: lower is better.
#    - Using quantile bins makes each bin have similar sample count, which is usually more stable
#      than uniform bins when probabilities are skewed.
#
# 5) Imbalanced problems:
#    - Accuracy can be misleading. Macro/weighted F1 are generally more informative.
#    - If you end up thresholding, tune threshold ONLY on validation, never on test.
#
# 6) Multiclass micro AUC + micro calibration:
#    - We compute micro by flattening one-hot and probs (treating each class logit as an instance).
#      This is a practical, robust approach across sklearn versions, but interpret it as
#      "overall ranking quality across all one-vs-rest decisions."
#
# 7) Regression correlations:
#    - Pearson/Spearman can be NaN if a target dimension is constant. That's a data issue worth surfacing.
#    - Keep the per-dimension metrics; averages can hide one target being terrible.
