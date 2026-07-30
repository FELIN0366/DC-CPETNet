# -*- coding: utf-8 -*-
"""Fig. 5a | Task-specific CPET variable-phase attribution."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from fig5_common import (
    SYSTEMS,
    TASK_LABELS,
    TASKS,
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
    write_csv,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=default_root())
    ap.add_argument("--out_dir", type=Path, default=default_out_dir())
    ap.add_argument("--top_n", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--window_fraction", type=float, default=0.8)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_pub_style()
    folds = discover_folds(args.root)
    variable_names = load_variable_names(folds)
    sys_idx = system_indices(variable_names)
    var_to_system = {
        variable_names[i]: system_name
        for system_name, idxs in sys_idx.items()
        for i in idxs
    }

    phase = np.linspace(0, 100, 200)
    rows = []
    display_rows = []
    y_labels = []
    y_task_centers = []
    group_boundaries = []

    for task in TASKS:
        stack = load_task_stack(folds, task, normalize_each_fold=True)
        mean_mat = stack.mean(axis=0)
        fold_var_imp = stack.mean(axis=1)
        var_imp = mean_mat.mean(axis=0)
        top_idx = np.argsort(var_imp)[::-1][: args.top_n]
        y_task_centers.append(len(display_rows) + (len(top_idx) - 1) / 2)

        for rank, idx in enumerate(top_idx, start=1):
            curve = mean_mat[:, idx]
            peak, left, right = high_window(curve, phase, args.window_fraction)
            display_rows.append(curve)
            y_labels.append(variable_names[idx])
            rows.append(
                {
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "rank": rank,
                    "variable": variable_names[idx],
                    "physiological_system": var_to_system.get(variable_names[idx], ""),
                    "mean_normalized_attribution": f"{var_imp[idx]:.8f}",
                    "fold_sd": f"{fold_var_imp[:, idx].std(ddof=1):.8f}",
                    "peak_phase_percent": f"{peak:.2f}",
                    "high_window_fraction": args.window_fraction,
                    "high_window_start_percent": f"{left:.2f}",
                    "high_window_end_percent": f"{right:.2f}",
                }
            )
        group_boundaries.append(len(display_rows) - 0.5)

    data = np.vstack(display_rows)
    data = p99_normalize(data)

    fig_h = max(5.6, 0.19 * data.shape[0] + 1.4)
    fig, ax = plt.subplots(figsize=(7.35, fig_h), constrained_layout=True)
    vivid_cmap = LinearSegmentedColormap.from_list(
        "cpet_vivid_attribution",
        ["#3E4BB8", "#37B7F2", "#58E0C2", "#DDF191", "#FDBA5B", "#E85A47"],
        N=256,
    )
    im = ax.imshow(
        data,
        aspect="auto",
        cmap=vivid_cmap,
        vmin=0,
        vmax=1,
        extent=[0, 100, data.shape[0] - 0.5, -0.5],
        interpolation="nearest",
    )

    for row_i, row in enumerate(rows):
        y = row_i
        left = float(row["high_window_start_percent"])
        right = float(row["high_window_end_percent"])
        peak = float(row["peak_phase_percent"])
        ax.hlines(y, left, right, color="#202020", linewidth=0.8)
        ax.plot(peak, y, marker="|", color="#202020", markersize=5.0, markeredgewidth=0.9)

    for boundary in group_boundaries[:-1]:
        ax.axhline(boundary, color="white", linewidth=1.5)

    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Normalized exercise phase (%)")

    ax_task = ax.twinx()
    ax_task.set_ylim(ax.get_ylim())
    ax_task.set_yticks(y_task_centers)
    ax_task.set_yticklabels([TASK_LABELS[t].replace(" | ", "\n") for t in TASKS], fontweight="bold")
    ax_task.tick_params(axis="y", length=0, pad=8)
    ax_task.spines["right"].set_visible(False)
    ax_task.spines["top"].set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.025)
    cbar.set_label("Normalized attribution")
    cbar.set_ticks([0, 0.5, 1.0])

    ax.text(
        99.5,
        -0.95,
        f"Black bars mark >= {int(args.window_fraction * 100)}% peak windows",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#333333",
    )

    ax.text(
        99.5,
        -0.95,
        f"Black bars mark >= {int(args.window_fraction * 100)}% peak windows",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#333333",
    )

    out_base = args.out_dir / "Fig5a_task_variable_phase_attribution"
    save_all(fig, out_base, dpi=args.dpi)
    plt.close(fig)

    write_csv(
        args.out_dir / "Fig5a_task_variable_phase_attribution.csv",
        rows,
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
    print(f"Saved Fig. 5a to {out_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
