# -*- coding: utf-8 -*-
"""Fig. 5a | Composite panel with task-level and class-level CPET attribution.

v3 updates:
1) restore two separate colorbars;
2) remove the Top1–Top3 explanatory legend text;
3) further reduce the gap between the two subpanels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import numpy as np

from fig5_common import (
    TASKS,
    TASK_LABELS,
    default_out_dir,
    default_root,
    discover_folds,
    high_window,
    load_task_stack,
    load_variable_names,
    p99_normalize,
    save_all,
    set_pub_style,
    system_indices,
    variable_to_system,
    write_csv,
)


CLASS_LABELS = {
    "t1": [(0, "A"), (1, "B"), (2, "C")],
    "t2": [(1, "Normal/\nnear-normal"), (0, "Mild–\nmoderate"), (2, "Severe–\nextreme")],
    "t3": [(0, "Positive"), (1, "Negative")],
    "t4": [(0, "Impaired"), (1, "Normal")],
    "t5": [(0, "Not\nexhausted"), (1, "Exhausted")],
}

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=default_root())
    ap.add_argument("--out_dir", type=Path, default=default_out_dir())
    ap.add_argument("--top_n_task", type=int, default=6)
    ap.add_argument("--top_n_class", type=int, default=3)
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


def load_class_mean(folds: Sequence[Path], task: str, class_id: int) -> tuple[np.ndarray, int]:
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


def build_left_task_panel(
    folds: Sequence[Path],
    variable_names: Sequence[str],
    top_n_task: int,
    window_fraction: float,
) -> tuple[np.ndarray, List[dict], List[str], List[dict]]:
    phase = np.linspace(0, 100, 200)
    display_rows = []
    row_meta = []
    y_labels = []
    csv_rows = []
    var_to_system = variable_to_system(variable_names)

    for ti, task in enumerate(TASKS):
        stack = load_task_stack(folds, task, normalize_each_fold=True)
        mean_mat = stack.mean(axis=0)
        fold_var_imp = stack.mean(axis=1)
        var_imp = mean_mat.mean(axis=0)
        top_idx = np.argsort(var_imp)[::-1][:top_n_task]
        base_row = ti * top_n_task

        for rank, idx in enumerate(top_idx, start=1):
            curve = mean_mat[:, idx]
            peak, left, right = high_window(curve, phase, window_fraction)
            display_rows.append(curve)
            y_labels.append(variable_names[idx])
            row_info = {
                "task": task,
                "task_label": TASK_LABELS[task],
                "rank": rank,
                "variable": variable_names[idx],
                "physiological_system": var_to_system.get(variable_names[idx], ""),
                "mean_normalized_attribution": float(var_imp[idx]),
                "fold_sd": float(fold_var_imp[:, idx].std(ddof=1)),
                "peak_phase_percent": float(peak),
                "high_window_fraction": float(window_fraction),
                "high_window_start_percent": float(left),
                "high_window_end_percent": float(right),
                "y_center": base_row + rank - 0.5,
            }
            row_meta.append(row_info)
            csv_rows.append(
                {
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "rank": rank,
                    "variable": variable_names[idx],
                    "physiological_system": var_to_system.get(variable_names[idx], ""),
                    "mean_normalized_attribution": f"{var_imp[idx]:.8f}",
                    "fold_sd": f"{fold_var_imp[:, idx].std(ddof=1):.8f}",
                    "peak_phase_percent": f"{peak:.2f}",
                    "high_window_fraction": window_fraction,
                    "high_window_start_percent": f"{left:.2f}",
                    "high_window_end_percent": f"{right:.2f}",
                }
            )

    data = np.vstack(display_rows)
    data = p99_normalize(data)
    return data, row_meta, y_labels, csv_rows


def build_right_class_panel(
    folds: Sequence[Path],
    variable_names: Sequence[str],
    top_n_class: int,
    window_fraction: float,
) -> tuple[List[dict], List[float], List[str], List[dict]]:
    phase_axis = np.linspace(0, 100, 200)
    task_boundaries = [0, 6, 12, 18, 24, 30]
    rows = []
    class_centers = []
    class_labels = []
    csv_rows = []

    for ti, task in enumerate(TASKS):
        y0 = task_boundaries[ti]
        y1 = task_boundaries[ti + 1]
        band_h = y1 - y0
        task_classes = CLASS_LABELS[task]
        class_h = band_h / len(task_classes)
        for ci, (class_id, class_label) in enumerate(task_classes):
            top = y0 + ci * class_h
            bottom = top + class_h
            center = (top + bottom) / 2.0
            mat, n_samples = load_class_mean(folds, task, class_id)
            var_imp = mat.mean(axis=0)
            top_idx = np.argsort(var_imp)[::-1][:top_n_class]
            cells = []
            for rank, idx in enumerate(top_idx, start=1):
                curve = mat[:, idx]
                peak, left, right = high_window(curve, phase_axis, window_fraction)
                cells.append(
                    {
                        "rank": rank,
                        "variable": variable_names[idx],
                        "peak": float(peak),
                        "start": float(left),
                        "end": float(right),
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
                        "high_window_fraction": window_fraction,
                        "high_window_start_percent": f"{left:.2f}",
                        "high_window_end_percent": f"{right:.2f}",
                    }
                )
            rows.append(
                {
                    "task": task,
                    "class_id": class_id,
                    "class_label": class_label,
                    "y_top": top,
                    "y_bottom": bottom,
                    "y_center": center,
                    "cells": cells,
                }
            )
            class_centers.append(center)
            class_labels.append(class_label)

    return rows, class_centers, class_labels, csv_rows


def main() -> None:
    args = parse_args()
    set_pub_style()
    folds = discover_folds(args.root)
    variable_names = load_variable_names(folds)
    _ = system_indices(variable_names)

    left_data, left_rows, left_y_labels, left_csv = build_left_task_panel(
        folds, variable_names, args.top_n_task, args.window_fraction
    )
    class_rows, class_centers, class_labels, right_csv = build_right_class_panel(
        folds, variable_names, args.top_n_class, args.window_fraction
    )

    attr_cmap = LinearSegmentedColormap.from_list(
        "cpet_vivid_attribution",
        ["#3E4BB8", "#37B7F2", "#58E0C2", "#DDF191", "#FDBA5B", "#E85A47"],
        N=256,
    )
    phase_cmap = LinearSegmentedColormap.from_list(
        "phase_vivid",
        ["#3E4BB8", "#37B7F2", "#58E0C2", "#DDF191", "#FDBA5B", "#E85A47"],
        N=256,
    )
    phase_norm = Normalize(vmin=0, vmax=100)

    fig = plt.figure(figsize=(10.0, 8.1), constrained_layout=True)
    gs = GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        height_ratios=[1.0, 0.04],
        width_ratios=[2.30, 1.58],
        wspace=0.02,
        hspace=0.03,
    )

    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    cax_left = fig.add_subplot(gs[1, 0])
    cax_right = fig.add_subplot(gs[1, 1])

    # --- Left panel ---
    im_left = ax_left.imshow(
        left_data,
        aspect="auto",
        cmap=attr_cmap,
        vmin=0,
        vmax=1,
        extent=[0, 100, left_data.shape[0], 0],
        interpolation="nearest",
    )

    for row in left_rows:
        y = row["y_center"]
        left = row["high_window_start_percent"]
        right = row["high_window_end_percent"]
        peak = row["peak_phase_percent"]
        ax_left.hlines(y, left, right, color="#202020", linewidth=0.75)
        ax_left.plot(peak, y, marker="o", color="#202020", markersize=2.5)

    for boundary in [6, 12, 18, 24]:
        ax_left.axhline(boundary, color="white", linewidth=2)

    ax_left.set_xlim(0, 100)
    ax_left.set_ylim(30, 0)
    ax_left.set_xticks([0, 20, 40, 60, 80, 100])
    ax_left.set_xlabel("Normalized exercise phase (%)")
    ax_left.set_yticks(np.arange(30) + 0.5)
    ax_left.set_yticklabels(left_y_labels)
    ax_left.tick_params(axis="y", pad=3)

    # --- Right class-level panel ---
    ax_right.set_xlim(0, args.top_n_class)
    ax_right.set_ylim(30, 0)
    ax_right.set_xticks(np.arange(args.top_n_class) + 0.5)
    ax_right.set_xticklabels([f"Top {i}" for i in range(1, args.top_n_class + 1)])
    ax_right.xaxis.tick_top()
    ax_right.tick_params(axis="x", length=0, pad=5)
    ax_right.set_yticks(class_centers)
    ax_right.set_yticklabels(class_labels)
    ax_right.yaxis.tick_right()
    ax_right.tick_params(axis="y", length=0, pad=6)
    for lab in ax_right.get_yticklabels():
        lab.set_fontsize(9)
    for lab in ax_right.get_xticklabels():
        lab.set_fontsize(9.0)
        lab.set_fontweight("bold")

    for row in class_rows:
        top = row["y_top"]
        bottom = row["y_bottom"]
        h = bottom - top
        for x, cell in enumerate(row["cells"]):
            peak = cell["peak"]
            ax_right.add_patch(
                Rectangle(
                    (x, top),
                    1,
                    h,
                    facecolor=phase_cmap(phase_norm(peak)),
                    edgecolor="white",
                    linewidth=1.0,
                )
            )
            color = text_color_for_phase(peak)
            ax_right.text(
                x + 0.5,
                top + 0.40 * h,
                cell["variable"],
                ha="center",
                va="center",
                fontsize=7.0,
                fontweight="semibold",
                color=color,
            )
            ax_right.text(
                x + 0.5,
                top + 0.71 * h,
                f"{peak:.0f}%",
                ha="center",
                va="center",
                fontsize=5.7,
                color=color,
            )

    for boundary in [6, 12, 18, 24]:
        ax_right.axhline(boundary, color="#333333", linewidth=2)
    for spine in ax_right.spines.values():
        spine.set_visible(False)

    # --- Colorbars ---
    cbar_left = fig.colorbar(im_left, cax=cax_left, orientation="horizontal")
    cbar_left.set_label("Normalized attribution")
    cbar_left.set_ticks([0, 0.5, 1.0])

    sm = plt.cm.ScalarMappable(norm=phase_norm, cmap=phase_cmap)
    cbar_right = fig.colorbar(sm, cax=cax_right, orientation="horizontal")
    cbar_right.set_label("Peak phase of attribution (%)")
    cbar_right.set_ticks([0, 50, 100])

    out_base = args.out_dir / "Fig5a_composite_task_and_class_attribution_revised_v3"
    save_all(fig, out_base, dpi=args.dpi)
    plt.close(fig)

    write_csv(
        args.out_dir / "Fig5a_composite_task_panel_left_revised_v3.csv",
        left_csv,
        [
            "task",
            "task_label",
            "rank",
            "variable",
            "physiological_system",
            "mean_normalized_attribution",
            "fold_sd",
            "peak_phase_percent",
            "high_window_fraction",
            "high_window_start_percent",
            "high_window_end_percent",
        ],
    )
    write_csv(
        args.out_dir / "Fig5a_composite_class_panel_right_revised_v3.csv",
        right_csv,
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
    print(f"Saved Fig. 5a composite to {out_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
