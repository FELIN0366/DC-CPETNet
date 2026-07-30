# -*- coding: utf-8 -*-
"""
Fig. Clinical interpretation 2 | PMGT nine-panel prior attention maps.

Input directory expected:
    ROOT/fold1/intermediates/t1_pmgt_attn.npy
    ROOT/fold1/intermediates/t6_pmgt_attn.npy
    ...

Also supports channel-name mapping from:
    ROOT/fold*/variable_time_attr/channel_name_mapping.json
or an explicitly provided file:
    --channel_mapping <path>

Supported attention shapes:
    [N, heads, 30, 30], [N, 30, 30], [heads, 30, 30], or [30, 30]
The script averages samples, heads, and folds.

Example:
    python plot_fig2_pmgt_prior_attention_v2_channel_mapping.py ^
      --root xx_path ^
      --out xx_path ^
      --channel_mapping xx_path
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

TASKS = ["t1", "t6"]
TASK_TITLES = {
    "t1": "t1 PMGT | Exercise cardiac function",
    "t6": "t6 PMGT | Disease-mechanism background",
}


def set_nature_style(font: str = "Arial") -> None:
    mpl.rcParams.update({
        "font.family": font,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    })


def discover_folds(root: Path) -> List[Path]:
    candidates = [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "intermediates").exists()]
    if not candidates and (root / "intermediates").exists():
        candidates = [root]
    if not candidates:
        raise FileNotFoundError(f"No fold folders containing intermediates found under: {root}")
    return candidates


def _mapping_dict_to_names(mapping: Dict, n_vars: int) -> List[str]:
    names = [None] * n_vars
    for k, v in mapping.items():
        try:
            idx = int(k)
        except Exception:
            continue
        if 0 <= idx < n_vars:
            names[idx] = str(v)
    if any(x is None for x in names):
        missing = [i for i, x in enumerate(names) if x is None]
        raise ValueError(f"channel mapping is incomplete, missing indices: {missing}")
    return names


def _try_load_names_from_json(p: Path, n_vars: int) -> Optional[List[str]]:
    obj = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        # Preferred: {"mapping": {"0": "MET", ...}}
        if "mapping" in obj and isinstance(obj["mapping"], dict):
            return _mapping_dict_to_names(obj["mapping"], n_vars)
        # Alternative flat dict of index -> name
        if all(str(k).isdigit() for k in obj.keys()):
            return _mapping_dict_to_names(obj, n_vars)
        # Alternative list-like fields
        for key in [
            "variable_names", "dynamic_variable_names", "cpet_variable_names",
            "dynamic_feature_names", "feature_names"
        ]:
            if key in obj and isinstance(obj[key], list):
                names = [str(x) for x in obj[key]]
                if len(names) >= n_vars:
                    return names[:n_vars]
        # Nested candidates (e.g., run_meta)
        for parent_key in ["meta", "config_summary", "preprocessing_summary", "dataset_meta"]:
            if parent_key in obj and isinstance(obj[parent_key], dict):
                nested = obj[parent_key]
                for key in [
                    "variable_names", "dynamic_variable_names", "cpet_variable_names",
                    "dynamic_feature_names", "feature_names"
                ]:
                    if key in nested and isinstance(nested[key], list):
                        names = [str(x) for x in nested[key]]
                        if len(names) >= n_vars:
                            return names[:n_vars]
    elif isinstance(obj, list):
        names = [str(x) for x in obj]
        if len(names) >= n_vars:
            return names[:n_vars]
    return None


def load_variable_names(root: Path, folds: List[Path], path: Optional[str], n_vars: int) -> List[str]:
    """
    优先级：
    1) --channel_mapping / --variable_names 显式指定文件
    2) fold*/variable_time_attr/channel_name_mapping.json
    3) root/variable_time_attr/channel_name_mapping.json
    4) fold*/run_meta.json 或 root/run_meta.json
    """
    candidates = []

    if path is not None:
        candidates.append(Path(path))

    for fd in folds:
        candidates.append(fd / "variable_time_attr" / "channel_name_mapping.json")
    candidates.append(root / "variable_time_attr" / "channel_name_mapping.json")
    candidates.append(root / "channel_name_mapping.json")

    for fd in folds:
        candidates.append(fd / "run_meta.json")
    candidates.append(root / "run_meta.json")

    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if not p.exists():
            continue
        try:
            if p.suffix.lower() == ".json":
                names = _try_load_names_from_json(p, n_vars)
                if names is not None:
                    print(f"[INFO] variable names loaded from: {p}")
                    return names
            else:
                names = []
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    names.append(line.split(",")[-1].strip() if "," in line else line)
                if len(names) >= n_vars:
                    print(f"[INFO] variable names loaded from: {p}")
                    return names[:n_vars]
        except Exception as e:
            print(f"[WARN] failed to parse variable names from {p}: {e}")

    raise ValueError(
        "未找到有效变量名映射。请提供 --channel_mapping，或确保 fold*/variable_time_attr/channel_name_mapping.json 存在且包含 mapping 字段。"
    )


def reduce_attention(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4:       # [N, H, V, V]
        return np.nanmean(arr, axis=(0, 1))
    if arr.ndim == 3:       # [N, V, V] or [H, V, V]
        return np.nanmean(arr, axis=0)
    if arr.ndim == 2:       # [V, V]
        return arr
    raise ValueError(f"Unsupported attention shape: {arr.shape}")


def load_task_attention(folds: List[Path], task: str, use_scores: bool = False) -> Dict[str, np.ndarray]:
    fold_maps = []
    for fd in folds:
        name = f"{task}_pmgt_attn_scores.npy" if use_scores else f"{task}_pmgt_attn.npy"
        f = fd / "intermediates" / name
        if not f.exists() and not use_scores:
            f = fd / "intermediates" / f"{task}_pmgt_attn_weights.npy"
        if not f.exists():
            print(f"[WARN] missing {name} in {fd}; skip")
            continue
        m = reduce_attention(np.load(f))
        fold_maps.append(m)
    if not fold_maps:
        raise FileNotFoundError(f"No attention arrays found for {task}")
    stack = np.stack(fold_maps, axis=0)
    return {
        "stack": stack,
        "mean": stack.mean(axis=0),
        "sem": stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0]) if stack.shape[0] > 1 else np.zeros_like(stack[0]),
    }


def maybe_remove_diag(m: np.ndarray) -> np.ndarray:
    x = m.copy()
    np.fill_diagonal(x, 0.0)
    return x


def top_edges_text(m: np.ndarray, names: List[str], k: int) -> str:
    x = maybe_remove_diag(m)
    flat_idx = np.argsort(-x.reshape(-1))[:k]
    lines = []
    n = x.shape[0]
    for idx in flat_idx:
        i, j = divmod(int(idx), n)
        lines.append(f"{names[j]} → {names[i]}: {x[i, j]:.4f}")
    return "\n".join(lines)


def plot_pmgt(root: Path, out: Path, variable_names_path: Optional[str], use_scores: bool,
              remove_diag: bool, cmap: str, top_k: int, dpi: int,
              cbar_left: float, cbar_width: float, subplot_wspace: float) -> None:
    folds = discover_folds(root)
    print("[INFO] folds:", [p.name for p in folds])
    data = {task: load_task_attention(folds, task, use_scores=use_scores) for task in TASKS}
    n_vars = data[TASKS[0]]["mean"].shape[0]
    names = load_variable_names(root, folds, variable_names_path, n_vars)

    maps = []
    for task in TASKS:
        m = data[task]["mean"]
        if remove_diag:
            m = maybe_remove_diag(m)
        maps.append(m)

    vmax = max(float(np.nanpercentile(m, 99)) for m in maps)
    vmin = min(float(np.nanpercentile(m, 1)) for m in maps) if use_scores else 0.0
    vmax = max(vmax, 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.0), constrained_layout=False)

    for ax, task, m in zip(axes, TASKS, maps):
        im = ax.imshow(m, aspect="equal", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(TASK_TITLES.get(task, task), pad=8)
        ax.set_xlabel("Source variable / key")
        ax.set_ylabel("Target variable / query")
        ax.set_xticks(np.arange(n_vars))
        ax.set_yticks(np.arange(n_vars))
        ax.set_xticklabels(names, rotation=90)
        ax.set_yticklabels(names)
        ax.tick_params(length=1.5)

    # 缩短两张子图距离，同时给右侧 colorbar 留位置
    fig.subplots_adjust(left=0.08, right=0.87, top=0.88, bottom=0.23, wspace=subplot_wspace)

    # 手动放置 colorbar，更容易继续往右移动
    cax = fig.add_axes([cbar_left, 0.14, cbar_width, 0.72])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Attention score" if use_scores else "Mean attention weight", labelpad=10)
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.yaxis.set_ticks_position("right")

    fig.suptitle("PMGT nine-panel-prior attention maps", y=0.98, fontsize=12, fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if out.suffix.lower() != ".pdf":
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")

    edge_report = []
    for task, m in zip(TASKS, maps):
        edge_report.append(f"[{task}]\n" + top_edges_text(m, names, k=top_k))
    out.with_suffix(".top_edges.txt").write_text("\n\n".join(edge_report), encoding="utf-8")
    print(f"[OK] saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=r"xx_path",
        help="Root containing fold1-fold5 interpretation output folders.",
    )
    parser.add_argument(
        "--out",
        default=r"xx_path",
        help="Output figure path.",
    )
    parser.add_argument("--variable_names", default=None, help="Optional variable-names file.")
    parser.add_argument("--channel_mapping", default=None, help="Preferred JSON mapping file, e.g. fold2/variable_time_attr/channel_name_mapping.json")
    parser.add_argument("--use_scores", action="store_true", help="Use *_attn_scores.npy instead of *_attn.npy")
    parser.add_argument("--keep_diag", action="store_true")
    parser.add_argument("--cmap", default="Purples")
    parser.add_argument("--font", default="Arial")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--cbar_left", type=float, default=0.89, help="Colorbar left position; larger means further right.")
    parser.add_argument("--cbar_width", type=float, default=0.018)
    parser.add_argument("--subplot_wspace", type=float, default=0.08, help="Horizontal gap between the two subplots.")
    args = parser.parse_args()

    set_nature_style(args.font)
    mapping_path = args.channel_mapping or args.variable_names
    plot_pmgt(
        Path(args.root),
        Path(args.out),
        mapping_path,
        args.use_scores,
        not args.keep_diag,
        args.cmap,
        args.top_k,
        args.dpi,
        args.cbar_left,
        args.cbar_width,
        args.subplot_wspace,
    )


if __name__ == "__main__":
    main()

