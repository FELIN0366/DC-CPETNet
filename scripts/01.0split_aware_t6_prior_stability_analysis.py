# -*- coding: utf-8 -*-
"""
Split-Aware T6 Prior Stability Analysis
========================================
用于评估 Experiment G: T6 Statistical Prior Injection 是否值得做。

核心逻辑：
1. 按 holdout_split_info_mtl.json 切分 dev / holdout
2. 在 dev 内部用与训练代码完全一致的 StratifiedKFold 重建 5-fold
3. 对每个 fold 统计 P(y_t | t6)，评估 train vs val / holdout 的分布稳定性
4. 运行 prior-only predictor，评估仅靠 t6 先验能达到的效果
5. 自动生成 recommend / caution / false 建议

重要约束：
- 只允许用 train prior 构建先验，val/holdout 统计仅用于稳定性分析
- 不加载任何模型 pth
- 不修改模型代码
"""

import argparse
import json
import os
import warnings
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

warnings.filterwarnings('ignore')

INVALID_VALUES = ['/', '-', '--', 'nan', 'NaN', 'None', '', ' ', '－']

TASK_COLS_DEFAULT = {
    "t1": "运动心功能分级",
    "t2": "运动耐量",
    "t3": "标准心电运动负荷试验",
    "t4": "运动中换气肺功能",
    "t5": "心率储备",
}

MERGED_COL_MAP = {
    "t2": "运动耐量",
}

GROUP_COL_CANDIDATES = ["匹配的第一大类", "匹配的大类", "临床疾病诊断", "t6"]


# ============================================================
# Data loading
# ============================================================

def _find_filename_col(df):
    """Find the filename column in the Excel."""
    for col in df.columns:
        cs = str(col)
        if '匹配' in cs and 'Excel' in cs and '文件' in cs:
            return col
    # Fallback: look for any column with .xlsx values
    for col in df.columns:
        sample = df[col].dropna().head(5).astype(str)
        if any('.xlsx' in v for v in sample):
            return col
    return None


def load_and_align_data(excel_path, split_info):
    """
    Load Excel and align with split info using filename matching.

    The holdout_split_info_mtl.json indices refer to positions in the
    sorted file list from data_root (after filtering), NOT to Excel rows.
    We use dev_filenames / test_filenames to match back to Excel rows.
    """
    df = pd.read_excel(excel_path, header=1)
    df = df.reset_index(drop=True)

    # Find the filename column
    fn_col = _find_filename_col(df)
    if fn_col is None:
        raise ValueError(

            "Cannot find filename column in Excel. "
            "Expected a column containing '匹配', 'Excel', '文件'."
        )
    print(f"  Filename column: '{fn_col}'")

    # Build filename -> Excel row index mapping
    fn_to_idx = {}
    for idx, fn in df[fn_col].items():
        if pd.notna(fn):
            fn_clean = str(fn).strip()
            fn_to_idx[fn_clean] = idx

    # Match split filenames to Excel rows
    dev_filenames = split_info["dev_filenames"]
    test_filenames = split_info["test_filenames"]

    dev_excel_indices = []
    for fn in dev_filenames:
        fn_clean = str(fn).strip()
        if fn_clean in fn_to_idx:
            dev_excel_indices.append(fn_to_idx[fn_clean])
        else:
            raise ValueError(f"Dev filename '{fn}' not found in Excel")

    test_excel_indices = []
    for fn in test_filenames:
        fn_clean = str(fn).strip()
        if fn_clean in fn_to_idx:
            test_excel_indices.append(fn_to_idx[fn_clean])
        else:
            raise ValueError(f"Test filename '{fn}' not found in Excel")

    assert len(dev_excel_indices) == split_info["n_dev"], (
        f"Dev size mismatch: matched {len(dev_excel_indices)}, "
        f"expected {split_info['n_dev']}"
    )
    assert len(test_excel_indices) == split_info["n_test"], (
        f"Holdout size mismatch: matched {len(test_excel_indices)}, "
        f"expected {split_info['n_test']}"
    )
    assert len(dev_excel_indices) + len(test_excel_indices) == split_info["n_samples"], (
        f"Total mismatch: {len(dev_excel_indices)} + {len(test_excel_indices)} "
        f"!= {split_info['n_samples']}"
    )

    # Now we need to map the dev_indices (positions in sorted file list)
    # back to Excel rows. The sorted file list order = sorted(dev_filenames + test_filenames).
    all_filenames_sorted = sorted(dev_filenames + test_filenames)
    # dev_indices[i] means: position in all_filenames_sorted -> that sample goes to dev
    # But the split_info already provides dev_filenames/test_filenames, so we just use those.

    # For KFold reconstruction, we need the mapping from "position in dev file list"
    # to "Excel row index". The dev file list = sorted(dev_filenames).
    dev_filenames_sorted = sorted(dev_filenames)
    dev_local_to_excel = {}
    for local_idx, fn in enumerate(dev_filenames_sorted):
        fn_clean = str(fn).strip()
        if fn_clean in fn_to_idx:
            dev_local_to_excel[local_idx] = fn_to_idx[fn_clean]

    dev_df = df.iloc[dev_excel_indices].copy()
    dev_df["_excel_idx"] = dev_excel_indices

    holdout_df = df.iloc[test_excel_indices].copy()
    holdout_df["_excel_idx"] = test_excel_indices

    return df, dev_df, holdout_df, dev_filenames_sorted, dev_local_to_excel, test_excel_indices


def resolve_group_col(df, preferred=None):
    """Resolve the group column name."""
    candidates = [preferred] if preferred else []
    candidates += [c for c in GROUP_COL_CANDIDATES if c not in candidates]
    for c in candidates:
        if c and c in df.columns:
            return c
    raise ValueError(
        f"Cannot find group column. Tried: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def resolve_task_cols(df, use_merged=False):
    """Resolve task column names, optionally using merged versions."""
    resolved = {}
    for task, col in TASK_COLS_DEFAULT.items():
        if use_merged and task in MERGED_COL_MAP:
            merged_col = MERGED_COL_MAP[task]
            if merged_col in df.columns:
                resolved[task] = merged_col
                continue
        if col in df.columns:
            resolved[task] = col
        else:
            print(f"  WARNING: task {task} column '{col}' not found, skipping")
    return resolved


def clean_series(series):
    """Clean a series by replacing invalid values with NaN."""
    s = series.copy()
    s = s.replace(INVALID_VALUES, np.nan)
    s = s.astype(str).str.strip()
    s = s.replace(['nan', 'None', 'NaN', ''], np.nan)
    return s


# ============================================================
# KFold reconstruction (must match training code exactly)
# ============================================================

def reconstruct_folds(dev_df, dev_filenames_sorted, dev_local_to_excel,
                      df, n_splits, kfold_seed, group_col):
    """
    Reconstruct 5-fold splits matching the training code exactly.

    Training code does:
        1. Load all .xlsx from data_root, sort them
        2. Filter by label availability and data loading success -> raw_datalist (sorted)
        3. dev_datalist = [raw_datalist[i] for i in dev_indices] (preserves sorted order)
        4. StratifiedKFold(n_splits=5, shuffle=True, random_state=3407)
           stratify = t6 label indices from dev_labellist

    The dev_filenames from split_info already represent the sorted dev file list.
    We reconstruct KFold on these sorted dev filenames, using t6 labels for stratification.

    Returns list of dicts with train/val indices in Excel row space.
    """
    n_dev = len(dev_filenames_sorted)

    # Get t6 labels for each dev sample (in sorted order)
    # dev_df may not be in sorted order, so we build a mapping
    fn_col = _find_filename_col(dev_df)
    fn_to_t6 = {}
    for _, row in dev_df.iterrows():
        fn = str(row[fn_col]).strip()
        t6_label = clean_series(pd.Series([row[group_col]]))[0]
        fn_to_t6[fn] = t6_label

    dev_labellist = []
    valid_mask = []
    for fn in dev_filenames_sorted:
        label = fn_to_t6.get(fn, np.nan)
        dev_labellist.append(label)
        valid_mask.append(pd.notna(label) and label != 'nan')

    # Create label mapping matching training code: sorted unique labels -> indices
    valid_labels = sorted(set(l for l in dev_labellist if pd.notna(l) and l != 'nan'))
    label_mapping = {v: i for i, v in enumerate(valid_labels)}
    label_indices = [label_mapping[l] for l in dev_labellist if pd.notna(l) and l != 'nan']

    # StratifiedKFold must match training code exactly
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=kfold_seed)

    # The training code does: splits = list(skf.split(dev_datalist, label_indices))
    # dev_datalist has n_dev items, label_indices has same length (only valid labels)
    # But wait — the training code filters out samples with None labels before splitting.
    # Let me check the actual code path more carefully.
    # In dataset_new.py, raw_labellist may contain None for files with no label.
    # The dev_labellist = [raw_labellist[i] for i in dev_indices] preserves all dev samples.
    # Then label_indices = [label_mapping[label] for label in dev_labellist]
    # This would fail if any label is None. So the code must filter before splitting.

    # Looking at the actual code flow: label_dict maps filename -> label,
    # and get_label_for_file returns None for unmapped files (which are skipped with `continue`).
    # So all entries in raw_labellist should have valid labels after filtering.
    # We assume all dev samples have valid t6 labels here.

    splits = list(skf.split(np.arange(n_dev), label_indices))

    fold_info = []
    for fold_idx, (train_local, val_local) in enumerate(splits):
        # Map local dev positions to Excel row indices
        train_excel_indices = [dev_local_to_excel[i] for i in train_local]
        val_excel_indices = [dev_local_to_excel[i] for i in val_local]

        fold_info.append({
            "fold": fold_idx,
            "train_local_indices": train_local.tolist(),
            "val_local_indices": val_local.tolist(),
            "train_indices_global": train_excel_indices,
            "val_indices_global": val_excel_indices,
            "train_size": len(train_local),
            "val_size": len(val_local),
        })

    return fold_info, label_mapping


# ============================================================
# Prior computation
# ============================================================

def compute_conditional_counts(df, group_col, task_col, global_indices=None):
    """
    Compute raw counts of (group, label) pairs.

    Returns:
        counts: dict[group][label] = count
        group_counts: dict[group] = total count
    """
    if global_indices is not None:
        sub = df.iloc[global_indices]
    else:
        sub = df

    g = clean_series(sub[group_col])
    t = clean_series(sub[task_col])

    counts = defaultdict(lambda: defaultdict(int))
    group_counts = defaultdict(int)

    for gi, ti in zip(g, t):
        if pd.isna(gi) or pd.isna(ti) or gi == 'nan' or ti == 'nan':
            continue
        counts[gi][ti] += 1
        group_counts[gi] += 1

    return dict(counts), dict(group_counts)


def get_all_labels(counts):
    """Get sorted union of all labels across all groups."""
    labels = set()
    for g_counts in counts.values():
        labels.update(g_counts.keys())
    return sorted(labels)


def compute_smoothed_probs(counts, group_counts, all_labels, alpha=1.0):
    """
    Compute Laplace-smoothed conditional probabilities P(y=c | t6=g).

    P(y=c | g) = (count(g,c) + alpha) / (count(g) + alpha * num_classes)
    """
    num_classes = len(all_labels)
    probs = {}
    log_probs = {}
    rare_groups = []

    for g in counts:
        n_g = group_counts.get(g, 0)
        if n_g < 5:
            rare_groups.append(g)

        probs[g] = {}
        log_probs[g] = {}
        for c in all_labels:
            count_gc = counts[g].get(c, 0)
            p = (count_gc + alpha) / (n_g + alpha * num_classes)
            probs[g][c] = p
            log_probs[g][c] = np.log(p + 1e-30)

    return probs, log_probs, rare_groups


def compute_global_prior(counts, group_counts, all_labels, alpha=1.0):
    """Compute global (marginal) prior P(y=c) across all groups."""
    num_classes = len(all_labels)
    total = 0
    label_totals = defaultdict(int)
    for g in counts:
        for c in all_labels:
            cnt = counts[g].get(c, 0)
            label_totals[c] += cnt
            total += cnt

    prior = {}
    log_prior = {}
    for c in all_labels:
        p = (label_totals[c] + alpha) / (total + alpha * num_classes)
        prior[c] = p
        log_prior[c] = np.log(p + 1e-30)

    return prior, log_prior


# ============================================================
# Stability metrics
# ============================================================

def total_variation_distance(p1, p2, labels):
    """TV = 0.5 * sum(|P1 - P2|)"""
    return 0.5 * sum(abs(p1.get(c, 0) - p2.get(c, 0)) for c in labels)


def js_divergence(p1, p2, labels):
    """Jensen-Shannon divergence."""
    v1 = np.array([p1.get(c, 1e-30) for c in labels])
    v2 = np.array([p2.get(c, 1e-30) for c in labels])
    v1 = v1 / v1.sum()
    v2 = v2 / v2.sum()
    return jensenshannon(v1, v2) ** 2


def max_abs_diff(p1, p2, labels):
    """max |P1 - P2|"""
    return max(abs(p1.get(c, 0) - p2.get(c, 0)) for c in labels)


def dominant_label(p_dist, labels):
    """Return the label with highest probability."""
    return max(labels, key=lambda c: p_dist.get(c, 0))


def compute_stability_metrics(train_probs, eval_probs, all_labels, groups):
    """
    Compute stability metrics between train and eval distributions.

    Returns per-group and aggregated metrics.
    """
    per_group = {}
    tv_list, js_list, mad_list = [], [], []
    consistency_count = 0
    total_groups = 0

    for g in groups:
        if g not in train_probs or g not in eval_probs:
            continue

        tv = total_variation_distance(train_probs[g], eval_probs[g], all_labels)
        js = js_divergence(train_probs[g], eval_probs[g], all_labels)
        mad = max_abs_diff(train_probs[g], eval_probs[g], all_labels)
        dom_train = dominant_label(train_probs[g], all_labels)
        dom_eval = dominant_label(eval_probs[g], all_labels)
        consistent = dom_train == dom_eval

        per_group[g] = {
            "TV": round(tv, 6),
            "JS": round(js, 6),
            "max_abs_diff": round(mad, 6),
            "dominant_train": dom_train,
            "dominant_eval": dom_eval,
            "dominant_consistent": consistent,
        }

        tv_list.append(tv)
        js_list.append(js)
        mad_list.append(mad)
        if consistent:
            consistency_count += 1
        total_groups += 1

    aggregated = {
        "mean_TV": round(np.mean(tv_list), 6) if tv_list else None,
        "mean_JS": round(np.mean(js_list), 6) if js_list else None,
        "mean_max_abs_diff": round(np.mean(mad_list), 6) if mad_list else None,
        "dominant_label_consistency": round(consistency_count / total_groups, 4) if total_groups > 0 else None,
    }

    return per_group, aggregated


# ============================================================
# Prior-only prediction
# ============================================================

def prior_only_predict(eval_df, train_probs, global_prior, group_col, task_col, all_labels):
    """
    Predict using only the train prior P(y | t6).

    For each sample, look up its t6 group in train_probs and predict argmax.
    If t6 group is unseen in train, fallback to global_prior.
    """
    g_series = clean_series(eval_df[group_col])
    t_series = clean_series(eval_df[task_col])

    y_true = []
    y_pred = []
    fallback_count = 0

    for gi, ti in zip(g_series, t_series):
        if pd.isna(ti) or ti == 'nan':
            continue
        y_true.append(ti)

        if pd.isna(gi) or gi == 'nan':
            pred = max(all_labels, key=lambda c: global_prior.get(c, 0))
            fallback_count += 1
        elif gi in train_probs:
            pred = max(all_labels, key=lambda c: train_probs[gi].get(c, 0))
        else:
            pred = max(all_labels, key=lambda c: global_prior.get(c, 0))
            fallback_count += 1

        y_pred.append(pred)

    if not y_true:
        return None

    labels_present = sorted(set(y_true) | set(y_pred))

    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 6),
        "macro_f1": round(f1_score(y_true, y_pred, labels=labels_present, average='macro', zero_division=0), 6),
        "weighted_f1": round(f1_score(y_true, y_pred, labels=labels_present, average='weighted', zero_division=0), 6),
        "precision_macro": round(precision_score(y_true, y_pred, labels=labels_present, average='macro', zero_division=0), 6),
        "recall_macro": round(recall_score(y_true, y_pred, labels=labels_present, average='macro', zero_division=0), 6),
        "n_samples": len(y_true),
        "fallback_count": fallback_count,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels_present).tolist(),
        "confusion_matrix_labels": labels_present,
    }

    return metrics


def majority_class_baseline(eval_df, task_col):
    """Compute majority-class baseline accuracy."""
    t = clean_series(eval_df[task_col]).dropna()
    t = t[t != 'nan']
    if len(t) == 0:
        return None
    majority = t.value_counts().index[0]
    acc = (t == majority).mean()
    return {"majority_class": majority, "accuracy": round(acc, 6)}


# ============================================================
# Recommendation engine
# ============================================================

def generate_recommendation(task, prior_only_metrics_folds, stability_folds, group_sample_sizes):
    """
    Auto-generate recommendation for each task.

    Rules:
    - recommend: prior-only holdout macro_f1 >> majority baseline, low JS, high consistency, adequate sample
    - caution: some drift, or marginal improvement, or sparse groups
    - false: prior-only ~ majority baseline, or inconsistent direction
    """
    holdout_f1s = [m["holdout"]["macro_f1"] for m in prior_only_metrics_folds if m["holdout"]]
    val_f1s = [m["val"]["macro_f1"] for m in prior_only_metrics_folds if m["val"]]
    holdout_accs = [m["holdout"]["accuracy"] for m in prior_only_metrics_folds if m["holdout"]]

    js_holdout = [s["train_vs_holdout"]["aggregated"]["mean_JS"]
                  for s in stability_folds
                  if s["train_vs_holdout"]["aggregated"].get("mean_JS") is not None]
    consistency_holdout = [s["train_vs_holdout"]["aggregated"]["dominant_label_consistency"]
                           for s in stability_folds
                           if s["train_vs_holdout"]["aggregated"].get("dominant_label_consistency") is not None]
    consistency_val = [s["train_vs_val"]["aggregated"]["dominant_label_consistency"]
                       for s in stability_folds
                       if s["train_vs_val"]["aggregated"].get("dominant_label_consistency") is not None]

    min_group_size = min(group_sample_sizes) if group_sample_sizes else 0
    sparse_groups = sum(1 for s in group_sample_sizes if s < 5)

    mean_holdout_f1 = np.mean(holdout_f1s) if holdout_f1s else 0
    mean_val_f1 = np.mean(val_f1s) if val_f1s else 0
    mean_js_holdout = np.mean(js_holdout) if js_holdout else 1.0
    mean_consistency_holdout = np.mean(consistency_holdout) if consistency_holdout else 0
    mean_consistency_val = np.mean(consistency_val) if consistency_val else 0

    # Majority baseline approximation: if holdout accuracy is very close to 1/num_classes
    # or if prior-only f1 is very low, it's likely near majority baseline
    mean_holdout_acc = np.mean(holdout_accs) if holdout_accs else 0

    # Heuristic thresholds
    f1_above_baseline = mean_holdout_f1 > 0.15  # non-trivial f1
    js_low = mean_js_holdout < 0.05
    js_moderate = mean_js_holdout < 0.15
    consistency_high = mean_consistency_holdout >= 0.8
    consistency_moderate = mean_consistency_holdout >= 0.6
    val_holdout_gap = mean_val_f1 - mean_holdout_f1
    no_severe_drift = val_holdout_gap < 0.15
    sparse_ratio = sparse_groups / len(group_sample_sizes) if group_sample_sizes else 1

    reasons = []

    if f1_above_baseline and js_low and consistency_high and sparse_ratio < 0.3:
        recommendation = "recommend"
        reasons.append(f"prior-only holdout macro_f1={mean_holdout_f1:.4f} (non-trivial)")
        reasons.append(f"train-vs-holdout JS={mean_js_holdout:.6f} (low drift)")
        reasons.append(f"dominant consistency={mean_consistency_holdout:.2%} (high)")
    elif (f1_above_baseline and js_moderate and consistency_moderate) or \
         (f1_above_baseline and (not no_severe_drift or sparse_ratio >= 0.3)):
        recommendation = "caution"
        if not no_severe_drift:
            reasons.append(f"val→holdout f1 gap={val_holdout_gap:.4f} (possible drift)")
        if sparse_ratio >= 0.3:
            reasons.append(f"{sparse_groups}/{len(group_sample_sizes)} groups have <5 samples")
        if not consistency_high:
            reasons.append(f"dominant consistency={mean_consistency_holdout:.2%} (moderate)")
        if not js_low:
            reasons.append(f"train-vs-holdout JS={mean_js_holdout:.6f} (moderate drift)")
    else:
        recommendation = "false"
        if not f1_above_baseline:
            reasons.append(f"prior-only holdout macro_f1={mean_holdout_f1:.4f} (~majority baseline)")
        if not js_moderate:
            reasons.append(f"train-vs-holdout JS={mean_js_holdout:.6f} (high drift)")
        if not consistency_moderate:
            reasons.append(f"dominant consistency={mean_consistency_holdout:.2%} (low)")

    return {
        "task": task,
        "recommend_prior_injection": recommendation,
        "mean_holdout_macro_f1": round(mean_holdout_f1, 6),
        "mean_val_macro_f1": round(mean_val_f1, 6),
        "mean_js_holdout": round(mean_js_holdout, 6),
        "mean_consistency_holdout": round(mean_consistency_holdout, 4),
        "min_group_size": min_group_size,
        "sparse_group_count": sparse_groups,
        "reasons": reasons,
    }


# ============================================================
# Main pipeline
# ============================================================

def run_analysis(args):
    """Run the full split-aware T6 prior stability analysis."""

    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Load split info ---
    with open(args.split_json, "r", encoding="utf-8") as f:
        split_info = json.load(f)

    print("=" * 70)
    print("Split-Aware T6 Prior Stability Analysis")
    print("=" * 70)
    print(f"  Input Excel : {args.input_excel}")
    print(f"  Split JSON  : {args.split_json}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Alpha       : {args.alpha}")
    print(f"  n_splits    : {args.n_splits}")
    print(f"  use_merged  : {args.use_merged_labels}")

    # --- 2. Load and align data ---
    print("\n[1/7] Loading and aligning data...")
    df, dev_df, holdout_df, dev_filenames_sorted, dev_local_to_excel, test_excel_indices = load_and_align_data(
        args.input_excel, split_info
    )
    print(f"  n_samples={split_info['n_samples']}, n_dev={len(dev_df)}, n_test={len(holdout_df)}")
    print("  Data alignment checks PASSED.")

    # --- 2.5. Save filtered data to Excel ---
    print("\n[Saving filtered data...]")
    filtered_excel_path = os.path.join(args.output_dir, "filtered_data_1149.xlsx")
    with pd.ExcelWriter(filtered_excel_path, engine="openpyxl") as writer:
        # Sheet 1: All 1149 samples
        all_filtered = pd.concat([dev_df, holdout_df], ignore_index=True)
        all_filtered.to_excel(writer, sheet_name="all_1149_samples", index=False)

        # Sheet 2: Dev set (920 samples)
        dev_df.to_excel(writer, sheet_name="dev_920_samples", index=False)

        # Sheet 3: Holdout set (229 samples)
        holdout_df.to_excel(writer, sheet_name="holdout_229_samples", index=False)

        # Sheet 4: Split info summary
        split_summary = pd.DataFrame([
            {"category": "total_matched", "count": 1149},
            {"category": "dev_set", "count": 920},
            {"category": "holdout_set", "count": 229},
            {"category": "excel_original_rows", "count": len(df)},
            {"category": "holdout_seed", "value": split_info["holdout_seed"]},
            {"category": "kfold_seed", "value": split_info["kfold_seed"]},
        ])
        split_summary.to_excel(writer, sheet_name="split_summary", index=False)

    print(f"  Saved filtered data to: {filtered_excel_path}")
    print(f"  - all_1149_samples: {len(all_filtered)} rows")
    print(f"  - dev_920_samples: {len(dev_df)} rows")
    print(f"  - holdout_229_samples: {len(holdout_df)} rows")

    # --- 3. Resolve columns ---
    group_col = resolve_group_col(df, preferred=args.group_col)
    task_cols = resolve_task_cols(df, use_merged=args.use_merged_labels)
    print(f"\n[2/7] Column resolution:")
    print(f"  group_col = '{group_col}'")
    for t, c in task_cols.items():
        print(f"  {t} -> '{c}'")

    # --- 4. Reconstruct folds ---
    print(f"\n[3/7] Reconstructing {args.n_splits}-fold splits (StratifiedKFold, seed={split_info['kfold_seed']})...")
    fold_info, t6_label_mapping = reconstruct_folds(
        dev_df, dev_filenames_sorted, dev_local_to_excel,
        df, args.n_splits, split_info["kfold_seed"], group_col
    )
    for fi in fold_info:
        print(f"  Fold {fi['fold']}: train={fi['train_size']}, val={fi['val_size']}, "
              f"holdout={len(holdout_df)}")

    # --- 5. Per-fold analysis ---
    all_fold_prior_tables = []
    all_fold_stability = []
    all_fold_prior_only = []
    recommendation_data = {}

    for fi in fold_info:
        fold_idx = fi["fold"]
        train_global = fi["train_indices_global"]
        val_global = fi["val_indices_global"]

        train_df = df.iloc[train_global]
        val_df = df.iloc[val_global]

        print(f"\n[4/7] Processing Fold {fold_idx} (train={len(train_df)}, val={len(val_df)}, holdout={len(holdout_df)})...")

        # ------ 5a. Compute priors per task ------
        prior_table = {
            "fold": fold_idx,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "holdout_size": len(holdout_df),
            "group_col": group_col,
            "task_cols": task_cols,
            "alpha": args.alpha,
            "tasks": {},
            "note": "train prior is the ONLY prior allowed for Experiment G; val/holdout stats are for stability analysis only",
        }

        stability_result = {
            "fold": fold_idx,
            "tasks": {},
        }

        prior_only_result = {
            "fold": fold_idx,
            "tasks": {},
        }

        for task, task_col in task_cols.items():
            print(f"    Task {task} ({task_col})...")

            # Compute counts for train, val, holdout
            train_counts, train_group_counts = compute_conditional_counts(
                df, group_col, task_col, train_global
            )
            val_counts, val_group_counts = compute_conditional_counts(
                df, group_col, task_col, val_global
            )
            holdout_counts, holdout_group_counts = compute_conditional_counts(
                df, group_col, task_col, test_excel_indices
            )

            # All labels (union across all splits)
            all_labels = sorted(
                set(get_all_labels(train_counts)) |
                set(get_all_labels(val_counts)) |
                set(get_all_labels(holdout_counts))
            )

            # Smoothed probabilities
            train_probs, train_log_probs, train_rare = compute_smoothed_probs(
                train_counts, train_group_counts, all_labels, args.alpha
            )
            val_probs, val_log_probs, _ = compute_smoothed_probs(
                val_counts, val_group_counts, all_labels, args.alpha
            )
            holdout_probs, holdout_log_probs, _ = compute_smoothed_probs(
                holdout_counts, holdout_group_counts, all_labels, args.alpha
            )

            # Global prior (from train only)
            global_prior, global_log_prior = compute_global_prior(
                train_counts, train_group_counts, all_labels, args.alpha
            )

            # Fallback: if val/holdout has groups not in train, use global prior
            val_groups_with_fallback = []
            holdout_groups_with_fallback = []
            all_groups = sorted(
                set(train_probs.keys()) | set(val_probs.keys()) | set(holdout_probs.keys())
            )

            for g in all_groups:
                if g not in train_probs:
                    train_probs[g] = dict(global_prior)
                    train_log_probs[g] = dict(global_log_prior)
                    if g in val_probs:
                        val_groups_with_fallback.append(g)
                    if g in holdout_probs:
                        holdout_groups_with_fallback.append(g)

            # Warnings
            warnings_list = []
            if train_rare:
                warnings_list.append(f"Rare train groups (<5 samples): {train_rare}")
            if val_groups_with_fallback:
                warnings_list.append(
                    f"Val groups not in train (using global prior fallback): {val_groups_with_fallback}"
                )
            if holdout_groups_with_fallback:
                warnings_list.append(
                    f"Holdout groups not in train (using global prior fallback): {holdout_groups_with_fallback}"
                )

            # Save prior table
            prior_table["tasks"][task] = {
                "task_col": task_col,
                "all_labels": all_labels,
                "all_groups": all_groups,
                "raw_counts_train": {g: dict(v) for g, v in train_counts.items()},
                "raw_counts_val": {g: dict(v) for g, v in val_counts.items()},
                "raw_counts_holdout": {g: dict(v) for g, v in holdout_counts.items()},
                "group_sample_sizes_train": train_group_counts,
                "group_sample_sizes_val": val_group_counts,
                "group_sample_sizes_holdout": holdout_group_counts,
                "smoothed_probs_train": train_probs,
                "smoothed_probs_val": val_probs,
                "smoothed_probs_holdout": holdout_probs,
                "log_probs_train": train_log_probs,
                "global_prior": global_prior,
                "global_log_prior": global_log_prior,
                "rare_group_warning": train_rare,
                "fallback_warnings": warnings_list,
            }

            # Stability metrics
            common_groups_tv = sorted(set(train_probs.keys()) & set(val_probs.keys()))
            common_groups_th = sorted(set(train_probs.keys()) & set(holdout_probs.keys()))

            tv_per_group_val, tv_agg_val = compute_stability_metrics(
                train_probs, val_probs, all_labels, common_groups_tv
            )
            tv_per_group_holdout, tv_agg_holdout = compute_stability_metrics(
                train_probs, holdout_probs, all_labels, common_groups_th
            )

            stability_result["tasks"][task] = {
                "train_vs_val": {
                    "per_group": tv_per_group_val,
                    "aggregated": tv_agg_val,
                },
                "train_vs_holdout": {
                    "per_group": tv_per_group_holdout,
                    "aggregated": tv_agg_holdout,
                },
            }

            # Prior-only prediction
            val_metrics = prior_only_predict(
                val_df, train_probs, global_prior, group_col, task_col, all_labels
            )
            holdout_metrics = prior_only_predict(
                holdout_df, train_probs, global_prior, group_col, task_col, all_labels
            )
            val_baseline = majority_class_baseline(val_df, task_col)
            holdout_baseline = majority_class_baseline(holdout_df, task_col)

            prior_only_result["tasks"][task] = {
                "val": val_metrics,
                "holdout": holdout_metrics,
                "val_majority_baseline": val_baseline,
                "holdout_majority_baseline": holdout_baseline,
            }

        # Save per-fold JSONs
        with open(os.path.join(args.output_dir, f"fold_{fold_idx}_prior_table_train.json"), "w", encoding="utf-8") as f:
            json.dump(prior_table, f, ensure_ascii=False, indent=2)

        with open(os.path.join(args.output_dir, f"fold_{fold_idx}_stability_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(stability_result, f, ensure_ascii=False, indent=2)

        with open(os.path.join(args.output_dir, f"fold_{fold_idx}_prior_only_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(prior_only_result, f, ensure_ascii=False, indent=2)

        all_fold_prior_tables.append(prior_table)
        all_fold_stability.append(stability_result)
        all_fold_prior_only.append(prior_only_result)

    # --- 6. Aggregate across folds ---
    print("\n[5/7] Aggregating across folds and generating recommendations...")

    for task in task_cols:
        task_stability = [s["tasks"][task] for s in all_fold_stability]
        task_prior_only = [p["tasks"][task] for p in all_fold_prior_only]

        # Collect group sample sizes from all folds
        group_sizes = []
        for pt in all_fold_prior_tables:
            task_data = pt["tasks"].get(task, {})
            gs = task_data.get("group_sample_sizes_train", {})
            group_sizes.extend(gs.values())

        rec = generate_recommendation(task, task_prior_only, task_stability, group_sizes)
        recommendation_data[task] = rec

    # --- 7. Summary outputs ---
    print("\n[6/7] Writing summary files...")

    # 7a. JSON summary
    summary_json = {
        "analysis_timestamp": datetime.now().isoformat(),
        "config": {
            "input_excel": args.input_excel,
            "split_json": args.split_json,
            "group_col": group_col,
            "task_cols": task_cols,
            "alpha": args.alpha,
            "n_splits": args.n_splits,
            "kfold_seed": split_info["kfold_seed"],
            "holdout_seed": split_info["holdout_seed"],
            "use_merged_labels": args.use_merged_labels,
        },
        "split_overview": {
            "n_samples": split_info["n_samples"],
            "n_dev": split_info["n_dev"],
            "n_test": split_info["n_test"],
            "folds": [{
                "fold": fi["fold"],
                "train_size": fi["train_size"],
                "val_size": fi["val_size"],
                "holdout_size": len(holdout_df),
            } for fi in fold_info],
        },
        "per_fold_stability": all_fold_stability,
        "per_fold_prior_only": all_fold_prior_only,
        "recommendations": recommendation_data,
    }

    summary_json_path = os.path.join(args.output_dir, "all_folds_prior_stability_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    # 7b. Excel summary
    excel_path = os.path.join(args.output_dir, "all_folds_prior_stability_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # Sheet: split_overview
        split_rows = []
        for fi in fold_info:
            split_rows.append({
                "fold": fi["fold"],
                "train_size": fi["train_size"],
                "val_size": fi["val_size"],
                "holdout_size": len(holdout_df),
            })
        pd.DataFrame(split_rows).to_excel(writer, sheet_name="split_overview", index=False)

        # Sheet: prior_counts_train
        counts_rows = []
        for pt in all_fold_prior_tables:
            fold = pt["fold"]
            for task, tdata in pt["tasks"].items():
                for g, gcounts in tdata.get("raw_counts_train", {}).items():
                    for label, cnt in gcounts.items():
                        counts_rows.append({
                            "fold": fold,
                            "task": task,
                            "group": g,
                            "label": label,
                            "train_count": cnt,
                            "train_group_total": tdata.get("group_sample_sizes_train", {}).get(g, 0),
                        })
        if counts_rows:
            pd.DataFrame(counts_rows).to_excel(writer, sheet_name="prior_counts_train", index=False)

        # Sheet: train_vs_val_stability
        stab_val_rows = []
        for fs in all_fold_stability:
            fold = fs["fold"]
            for task, tdata in fs["tasks"].items():
                agg = tdata.get("train_vs_val", {}).get("aggregated", {})
                stab_val_rows.append({
                    "fold": fold,
                    "task": task,
                    "mean_TV": agg.get("mean_TV"),
                    "mean_JS": agg.get("mean_JS"),
                    "mean_max_abs_diff": agg.get("mean_max_abs_diff"),
                    "dominant_label_consistency": agg.get("dominant_label_consistency"),
                })
        if stab_val_rows:
            pd.DataFrame(stab_val_rows).to_excel(writer, sheet_name="train_vs_val_stability", index=False)

        # Sheet: train_vs_holdout_stability
        stab_ho_rows = []
        for fs in all_fold_stability:
            fold = fs["fold"]
            for task, tdata in fs["tasks"].items():
                agg = tdata.get("train_vs_holdout", {}).get("aggregated", {})
                stab_ho_rows.append({
                    "fold": fold,
                    "task": task,
                    "mean_TV": agg.get("mean_TV"),
                    "mean_JS": agg.get("mean_JS"),
                    "mean_max_abs_diff": agg.get("mean_max_abs_diff"),
                    "dominant_label_consistency": agg.get("dominant_label_consistency"),
                })
        if stab_ho_rows:
            pd.DataFrame(stab_ho_rows).to_excel(writer, sheet_name="train_vs_holdout_stability", index=False)

        # Sheet: prior_only_val_metrics
        po_val_rows = []
        for fp in all_fold_prior_only:
            fold = fp["fold"]
            for task, tdata in fp["tasks"].items():
                m = tdata.get("val") or {}
                bl = tdata.get("val_majority_baseline") or {}
                po_val_rows.append({
                    "fold": fold,
                    "task": task,
                    "accuracy": m.get("accuracy"),
                    "macro_f1": m.get("macro_f1"),
                    "weighted_f1": m.get("weighted_f1"),
                    "precision_macro": m.get("precision_macro"),
                    "recall_macro": m.get("recall_macro"),
                    "majority_baseline_acc": bl.get("accuracy"),
                    "fallback_count": m.get("fallback_count"),
                })
        if po_val_rows:
            pd.DataFrame(po_val_rows).to_excel(writer, sheet_name="prior_only_val_metrics", index=False)

        # Sheet: prior_only_holdout_metrics
        po_ho_rows = []
        for fp in all_fold_prior_only:
            fold = fp["fold"]
            for task, tdata in fp["tasks"].items():
                m = tdata.get("holdout") or {}
                bl = tdata.get("holdout_majority_baseline") or {}
                po_ho_rows.append({
                    "fold": fold,
                    "task": task,
                    "accuracy": m.get("accuracy"),
                    "macro_f1": m.get("macro_f1"),
                    "weighted_f1": m.get("weighted_f1"),
                    "precision_macro": m.get("precision_macro"),
                    "recall_macro": m.get("recall_macro"),
                    "majority_baseline_acc": bl.get("accuracy"),
                    "fallback_count": m.get("fallback_count"),
                })
        if po_ho_rows:
            pd.DataFrame(po_ho_rows).to_excel(writer, sheet_name="prior_only_holdout_metrics", index=False)

        # Sheet: recommendation_summary
        rec_rows = []
        for task, rec in recommendation_data.items():
            rec_rows.append({
                "task": task,
                "recommend_prior_injection": rec["recommend_prior_injection"],
                "mean_holdout_macro_f1": rec["mean_holdout_macro_f1"],
                "mean_val_macro_f1": rec["mean_val_macro_f1"],
                "mean_js_holdout": rec["mean_js_holdout"],
                "mean_consistency_holdout": rec["mean_consistency_holdout"],
                "min_group_size": rec["min_group_size"],
                "sparse_group_count": rec["sparse_group_count"],
                "reasons": "; ".join(rec["reasons"]),
            })
        pd.DataFrame(rec_rows).to_excel(writer, sheet_name="recommendation_summary", index=False)

    print(f"  Summary JSON: {summary_json_path}")
    print(f"  Summary Excel: {excel_path}")

    # --- 8. Console summary ---
    print("\n" + "=" * 70)
    print("FINAL CONSOLE SUMMARY")
    print("=" * 70)

    print("\n[Data Alignment]")
    print(f"  Excel total rows: {len(df)}")
    print(f"  Matched samples:  {len(dev_df) + len(holdout_df)} == {split_info['n_samples']} -> {'PASS' if len(dev_df) + len(holdout_df) == split_info['n_samples'] else 'FAIL'}")
    print(f"  n_dev match:      {len(dev_df)} == {split_info['n_dev']} -> {'PASS' if len(dev_df) == split_info['n_dev'] else 'FAIL'}")
    print(f"  n_test match:     {len(holdout_df)} == {split_info['n_test']} -> {'PASS' if len(holdout_df) == split_info['n_test'] else 'FAIL'}")

    print("\n[Fold Sample Sizes]")
    for fi in fold_info:
        print(f"  Fold {fi['fold']}: train={fi['train_size']}, val={fi['val_size']}, holdout={len(holdout_df)}")

    print("\n[Prior-Only Holdout macro_f1 (mean +/- std across folds)]")
    for task in task_cols:
        f1s = []
        for fp in all_fold_prior_only:
            m = fp["tasks"].get(task, {}).get("holdout")
            if m:
                f1s.append(m["macro_f1"])
        if f1s:
            print(f"  {task}: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        else:
            print(f"  {task}: N/A")

    print("\n[Train-vs-Holdout Average JS Divergence]")
    for task in task_cols:
        js_vals = []
        for fs in all_fold_stability:
            agg = fs["tasks"].get(task, {}).get("train_vs_holdout", {}).get("aggregated", {})
            if agg.get("mean_JS") is not None:
                js_vals.append(agg["mean_JS"])
        if js_vals:
            print(f"  {task}: {np.mean(js_vals):.6f}")
        else:
            print(f"  {task}: N/A")

    print("\n[Experiment G Recommendation]")
    for task, rec in recommendation_data.items():
        tag = rec["recommend_prior_injection"].upper()
        if tag == "RECOMMEND":
            label = "recommend"
        elif tag == "CAUTION":
            label = "caution"
        else:
            label = "reject"
        print(f"  {task}: {label}")
        for r in rec["reasons"]:
            print(f"         - {r}")

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Split-Aware T6 Prior Stability Analysis for Experiment G"
    )
    parser.add_argument(
        "--input_excel", type=str,
        default=r"xx_path",
        help="Path to the merged Excel file"
    )
    parser.add_argument(
        "--split_json", type=str,
        default=r"xx_path",
        help="Path to holdout_split_info_mtl.json"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=r"xx_path",
        help="Output directory"
    )
    parser.add_argument(
        "--group_col", type=str, default=None,
        help="T6 group column name (default: auto-detect, prefer '匹配的第一大类')"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Laplace smoothing alpha (default: 1.0)"
    )
    parser.add_argument(
        "--n_splits", type=int, default=5,
        help="Number of CV folds (default: 5)"
    )
    parser.add_argument(
        "--use_merged_labels", action="store_true", default=False,
        help="Use merged label columns where available (e.g. 运动耐量_合并)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args)

