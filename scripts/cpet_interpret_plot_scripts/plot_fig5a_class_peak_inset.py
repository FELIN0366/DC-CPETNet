# -*- coding: utf-8 -*-
"""Class-level top-variable and peak-phase summary for Fig. 5a.

This compact inset summarizes class-specific attribution without drawing full
per-class heatmaps. Each cell reports one top variable and its peak phase.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np

from fig5_common import (
    TASKS,
    default_out_dir,
    default_root,
    discover_folds,
    high_window,
    load_variable_names,
    p99_normalize,
    save_all,
    set_pub_style,
    write_csv,
)


CLASS_LABELS = {
    "t1": [(0, "A"), (1, "B"), (2, "C")],
    "t2": [
        (1, "Normal"),
        (0, "Mild-mod"),
        (2, "Severe-ext"),
    ],
    "t3": [(0, "Positive"), (1, "Negative")],
    "t4": [(0, "Impaired"), (1, "Normal")],
    "t5": [(0, "Not exh."), (1, "Exh.")],
}

TASK_NAMES = {
    "t1": "t1",
    "t2": "t2",
    "t3": "t3",
    "t4": "t4",
    "t5": "t5",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=default_root())
    ap.add_argument("--out_dir", type=Path, default=default_out_dir())
    ap.add_argument("--top_n", type=int, default=3)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--window_fraction", type=float, default=0.8)
    return ap.parse_args()


def orient_sample_time_variable(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim != 3:
        raise ValueError(f"Expected 3D sample-time-variable attribution array, got {a.shape}")
    if a.shape[1:] == (200, 30):
        return a
    if a.shape[1:] == (30, 200):
        return np.transpose(a, (0, 2, 1))
    raise ValueError(f"Cannot orient sample attribution array with shape {a.shape}")


def load_class_mean(folds, task: str, class_id: int) -> tuple[np.ndarray, int]:
    class_mats = []
    class_counts = []
    for fold in folds:
        attr_path = fold / "variable_time_attr" / f"{task}_all_samples.npy"
        label_path = fold / "context_counterfactual" / f"labels_{task}.npy"
        if not attr_path.exists():
            raise FileNotFoundError(f"Missing attribution file: {attr_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing class label file: {label_path}")
        attr = orient_sample_time_variable(np.load(attr_path))
        labels = np.load(label_path)
        if attr.shape[0] != labels.shape[0]:
            raise ValueError(f"Sample mismatch for {task}: {attr.shape[0]} attr vs {labels.shape[0]} labels")
        mask = labels == class_id
        if not np.any(mask):
            continue
        class_counts.append(int(mask.sum()))
        class_mats.append(p99_normalize(attr[mask].mean(axis=0)))
    if not class_mats:
        raise ValueError(f"No samples found for {task} class {class_id}")
    return np.stack(class_mats, axis=0).mean(axis=0), int(round(float(np.mean(class_counts))))


def text_color_for_phase(phase: float) -> str:
    return "white" if phase < 14 or phase > 84 else "#1D1D1D"


def main() -> None:
    args = parse_args()
    set_pub_style()
    folds = discover_folds(args.root)
    variable_names = load_variable_names(folds)
    phase_axis = np.linspace(0, 100, 200)

    rows = []
    csv_rows = []
    for task in TASKS:
        for class_id, class_label in CLASS_LABELS[task]:
            mat, n_samples = load_class_mean(folds, task, class_id)
            var_imp = mat.mean(axis=0)
            top_idx = np.argsort(var_imp)[::-1][: args.top_n]
            cell_values = []
            for rank, idx in enumerate(top_idx, start=1):
                curve = mat[:, idx]
                peak, left, right = high_window(curve, phase_axis, args.window_fraction)
                cell_values.append(
                    {
                        "rank": rank,
                        "variable": variable_names[idx],
                        "peak": peak,
                        "start": left,
                        "end": right,
                        "importance": float(var_imp[idx]),
                    }
                )
                csv_rows.append(
                    {
                        "task": task,
                        "class_id": class_id,
                        "class_label": class_label,
                        "n_samples_per_fold": n_samples,
                        "rank": rank,
                        "variable": variable_names[idx],
                        "mean_normalized_attribution": f"{var_imp[idx]:.8f}",
                        "peak_phase_percent": f"{peak:.2f}",
                        "high_window_fraction": args.window_fraction,
                        "high_window_start_percent": f"{left:.2f}",
                        "high_window_end_percent": f"{right:.2f}",
                    }
                )
            rows.append(
                {
                    "task": task,
                    "class_id": class_id,
                    "class_label": class_label,
                    "cells": cell_values,
                }
            )

    cmap = LinearSegmentedColormap.from_list(
        "phase_vivid",
        ["#3E4BB8", "#37B7F2", "#58E0C2", "#DDF191", "#FDBA5B", "#E85A47"],
        N=256,
    )
    norm = Normalize(vmin=0, vmax=100)

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(3.65, 4.35), constrained_layout=True)
    ax.set_xlim(0, args.top_n)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(args.top_n) + 0.5)
    ax.set_xticklabels([f"Top {i}" for i in range(1, args.top_n + 1)])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([f"{r['task']}-{r['class_label']}" for r in rows])
    ax.tick_params(axis="y", length=0, pad=3)

    for y, row in enumerate(rows):
        for x, cell in enumerate(row["cells"]):
            peak = cell["peak"]
            ax.add_patch(
                Rectangle(
                    (x, y),
                    1,
                    1,
                    facecolor=cmap(norm(peak)),
                    edgecolor="white",
                    linewidth=1.0,
                )
            )
            ax.text(
                x + 0.5,
                y + 0.43,
                cell["variable"],
                ha="center",
                va="center",
                fontsize=6.0,
                fontweight="bold" if cell["rank"] == 1 else "normal",
                color=text_color_for_phase(peak),
            )
            ax.text(
                x + 0.5,
                y + 0.69,
                f"{peak:.0f}%",
                ha="center",
                va="center",
                fontsize=5.7,
                color=text_color_for_phase(peak),
            )

    boundary = 0
    for task in TASKS[:-1]:
        boundary += len(CLASS_LABELS[task])
        ax.axhline(boundary, color="#333333", linewidth=0.7)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        orientation="horizontal",
        fraction=0.055,
        pad=0.035,
    )
    cbar.set_label("Peak exercise phase (%)")
    cbar.set_ticks([0, 50, 100])

    out_base = args.out_dir / "Fig5a_class_peak_inset"
    save_all(fig, out_base, dpi=args.dpi)
    plt.close(fig)

    write_csv(
        args.out_dir / "Fig5a_class_peak_inset.csv",
        csv_rows,
        [
            "task",
            "class_id",
            "class_label",
            "n_samples_per_fold",
            "rank",
            "variable",
            "mean_normalized_attribution",
            "peak_phase_percent",
            "high_window_fraction",
            "high_window_start_percent",
            "high_window_end_percent",
        ],
    )
    print(f"Saved Fig. 5a class inset to {out_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
