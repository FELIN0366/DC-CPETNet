# -*- coding: utf-8 -*-
"""Fig. 5b | Task-level attribution summarized by physiological system.

Revised version:
1. Uses a tighter, publication-style layout.
2. Adds S0-S3 identifiers to the physiological-system labels.
3. Uses a more informative attribution range (default vmax=0.30).
4. Highlights each task's dominant system with a thin system-colored outline
   and bold numeric annotation.
5. Uses the unified label "Mean normalized attribution".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
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


# ============================================================
# Manually editable axis-label maps
# Use "\n" wherever you want a line break.
# ============================================================

TASK_TICK_LABELS = {
    "t1": "t1\nCPET functional\nclass",
    "t2": "t2\nExercise\ncapacity",
    "t3": "t3\nExercise ECG\ninterpretation",
    "t4": "t4\nVentilatory\nfunction",
    "t5": "t5\nHeart-rate\nreserve",
}

SYSTEM_TICK_LABELS = {
    "S0 Oxygen delivery": "S0\nOxygen\ndelivery",
    "S1 Ventilatory drive": "S1\nVentilatory\ndrive",
    "S2 Gas-exchange efficiency": "S2\nGas-exchange\nefficiency",
    "S3 Reserve / stability": "S3\nReserve /\nstability",
}


def _bold_first_line(label: str) -> str:
    """Bold the first line of a manually defined multiline label."""
    lines = label.split("\n")

    if len(lines) == 1:
        return rf"$\bf{{{lines[0]}}}$"

    return rf"$\bf{{{lines[0]}}}$" + "\n" + "\n".join(lines[1:])


def _task_tick_label(task: str) -> str:
    return _bold_first_line(TASK_TICK_LABELS[task])


def _system_tick_label(system_name: str) -> str:
    return _bold_first_line(SYSTEM_TICK_LABELS[system_name])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=default_root())
    ap.add_argument("--out_dir", type=Path, default=default_out_dir())
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument(
        "--vmax",
        type=float,
        default=0.30,
        help=(
            "Upper limit of the attribution color scale. The actual upper limit "
            "is automatically increased if the matrix contains a larger value."
        ),
    )
    ap.add_argument(
        "--output_name",
        type=str,
        default="Fig5b_task_system_attribution",
        help="Base filename used for PNG/PDF/SVG and CSV outputs.",
    )
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

    # Keep one fixed scale across all task-system cells while avoiding clipping.
    plot_vmax = max(float(args.vmax), float(mat.max()) * 1.03)

    # Sequential attribution palette. Categorical S0-S3 colors are reserved for
    # the top strips and row-wise maximum outlines.
    cmap = LinearSegmentedColormap.from_list(
        "clinical_attribution",
        ["#F7F9FB", "#DCE7F0", "#A9C5DA", "#6F9CBE", "#376F99"],
        N=256,
    )

    # Manual layout gives more predictable spacing than constrained_layout for
    # long multiline labels and an external colorbar.
    fig = plt.figure(figsize=(4.25, 3.20))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.0, 0.042],
        left=0.34,
        right=0.93,
        bottom=0.25,
        top=0.91,
        wspace=0.08,
    )
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    im = ax.imshow(
        mat,
        cmap=cmap,
        vmin=0,
        vmax=plot_vmax,
        aspect="equal",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(system_names)))
    ax.set_xticklabels(
        [_system_tick_label(name) for name in system_names],
        fontsize=6,
        linespacing=0.95,
    )
    ax.set_yticks(np.arange(len(TASKS)))
    ax.set_yticklabels(
        [_task_tick_label(task) for task in TASKS],
        fontsize=6,
        linespacing=0.95,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", length=0, pad=7)

    # Cell values and row-wise dominant-system indication.
    for i in range(mat.shape[0]):
        row_max = int(np.argmax(mat[i]))

        for j in range(mat.shape[1]):
            is_max = j == row_max
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7.0,
                fontweight="bold" if is_max else "normal",
                color="#111111",
            )

        ax.add_patch(
            patches.Rectangle(
                (row_max - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor=SYSTEM_COLORS[system_names[row_max]],
                linewidth=1.10,
                zorder=4,
            )
        )

    # System-color strips establish a direct visual link to Fig. 5c.
    for j, system_name in enumerate(system_names):
        ax.add_patch(
            patches.Rectangle(
                (j - 0.5, -0.67),
                1,
                0.075,
                facecolor=SYSTEM_COLORS[system_name],
                edgecolor="none",
                clip_on=False,
                zorder=5,
            )
        )

    # Light cell boundaries retain matrix structure without overpowering values.
    ax.set_xticks(np.arange(-0.5, len(system_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(TASKS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Mean normalized attribution", labelpad=6)
    cbar.set_ticks([0.0, plot_vmax / 2.0, plot_vmax])
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar.ax.tick_params(length=2.5, pad=3)
    cbar.outline.set_linewidth(0.75)

    out_base = args.out_dir / args.output_name
    save_all(fig, out_base, dpi=args.dpi)
    plt.close(fig)

    write_csv(
        args.out_dir / f"{args.output_name}.csv",
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
