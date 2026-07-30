# -*- coding: utf-8 -*-
"""Fig. 5b | Task-level attribution summarized by physiological system."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from fig5_common import (
    SYSTEMS,
    SYSTEM_COLORS,
    TASKS,
    default_out_dir,
    default_root,
    discover_folds,
    load_task_stack,
    load_variable_names,
    save_all,
    set_pub_style,
    system_indices,
    write_csv,
)

# =========================
# Axis label mappings
# Modify "\n" to control line breaks
# =========================

X_LABEL_MAP = {
    "S0 Oxygen delivery": "S0\nOxygen\ndelivery",
    "S1 Ventilatory drive": "S1\nVentilatory\ndrive",
    "S2 Gas-exchange efficiency": "S2\nGas-exchange\nefficiency",
    "S3 Reserve / stability": "S3\nReserve\nstability",
}

Y_LABEL_MAP = {
    "t1": "t1\nCPET\nfunctional\nclass",
    "t2": "t2\nExercise\ncapacity",
    "t3": "t3\nExercise ECG\ninterpretation",
    "t4": "t4\nVentilatory\nfunction",
    "t5": "t5\nHeart-rate\nreserve",
}

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=default_root())
    ap.add_argument("--out_dir", type=Path, default=default_out_dir())
    ap.add_argument("--dpi", type=int, default=600)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_pub_style()
    folds = discover_folds(args.root)
    variable_names = load_variable_names(folds)
    sys_idx = system_indices(variable_names)
    system_names = [name for name, _ in SYSTEMS]

    mat = np.zeros((len(TASKS), len(system_names)), dtype=float)
    sd = np.zeros_like(mat)
    rows = []

    for ti, task in enumerate(TASKS):
        stack = load_task_stack(folds, task, normalize_each_fold=True)
        for si, system_name in enumerate(system_names):
            idxs = sys_idx[system_name]
            fold_values = stack[:, :, idxs].mean(axis=(1, 2))
            mat[ti, si] = fold_values.mean()
            sd[ti, si] = fold_values.std(ddof=1)
            rows.append(
                {
                    "task": task,
                    "system": system_name,
                    "mean_normalized_attribution": f"{mat[ti, si]:.8f}",
                    "fold_sd": f"{sd[ti, si]:.8f}",
                    "n_variables": len(idxs),
                    "variables": "; ".join(variable_names[i] for i in idxs),
                }
            )

    cmap = LinearSegmentedColormap.from_list(
        "clinical_attribution",
        ["#F7F9FB", "#D7E4EF", "#8FB6D4", "#3E759F"],
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=max(0.5, float(mat.max()) * 1.08), aspect="auto")

    ax.set_xticks(np.arange(len(system_names)))
    ax.set_xticklabels(
        [X_LABEL_MAP[name] for name in system_names],
        fontsize=6.0,
        linespacing=0.95,
    )

    ax.set_yticks(np.arange(len(TASKS)))
    ax.set_yticklabels(
        [Y_LABEL_MAP[task] for task in TASKS],
        fontsize=6.0,
        linespacing=0.95,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", length=0)

    for i in range(mat.shape[0]):
        row_max = int(np.argmax(mat[i]))
        for j in range(mat.shape[1]):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6.7,
                fontweight="bold" if j == row_max else "normal",
            )
        ax.add_patch(
            patches.Rectangle(
                (row_max - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor=SYSTEM_COLORS[system_names[row_max]],
                linewidth=2,
            )
        )

    for j, system_name in enumerate(system_names):
        ax.add_patch(
            patches.Rectangle(
                (j - 0.5, -0.64),
                1,
                0.08,
                facecolor=SYSTEM_COLORS[system_name],
                edgecolor="none",
                clip_on=False,
            )
        )

    ax.set_xticks(np.arange(-0.5, len(system_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(TASKS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Mean attribution")
    cbar.set_ticks([0, 0.25, 0.5])

    out_base = args.out_dir / "Fig5b_task_system_attribution"
    save_all(fig, out_base, dpi=args.dpi)
    plt.close(fig)

    write_csv(
        args.out_dir / "Fig5b_task_system_attribution.csv",
        rows,
        [
            "task",
            "system",
            "mean_normalized_attribution",
            "fold_sd",
            "n_variables",
            "variables",
        ],
    )
    print(f"Saved Fig. 5b to {out_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
