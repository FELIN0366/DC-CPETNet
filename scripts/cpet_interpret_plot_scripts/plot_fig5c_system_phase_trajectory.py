# -*- coding: utf-8 -*-
"""Fig. 5c | Shared phase trajectory by physiological system."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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

    phase = np.linspace(0, 100, 200)
    trajectories = {}
    rows = []

    for system_name in system_names:
        curves = []
        for task in TASKS:
            stack = load_task_stack(folds, task, normalize_each_fold=True)
            idxs = sys_idx[system_name]
            task_fold_curves = stack[:, :, idxs].mean(axis=2)
            curves.append(task_fold_curves)
        all_curves = np.concatenate(curves, axis=0)
        mean = all_curves.mean(axis=0)
        sem = all_curves.std(axis=0, ddof=1) / np.sqrt(all_curves.shape[0])
        trajectories[system_name] = (mean, sem)
        for p, m, s in zip(phase, mean, sem):
            rows.append(
                {
                    "phase_percent": f"{p:.4f}",
                    "system": system_name,
                    "mean_normalized_attribution": f"{m:.8f}",
                    "sem_across_task_fold_curves": f"{s:.8f}",
                    "n_task_fold_curves": all_curves.shape[0],
                }
            )

    fig, ax = plt.subplots(figsize=(6.45, 3.55), constrained_layout=True)
    ax.axvspan(0, 20, color="#F4F4F4", zorder=0)
    ax.axvspan(60, 100, color="#F8F1E8", zorder=0)
    ax.text(
        10,
        0.045,
        "Early",
        ha="center",
        va="bottom",
        transform=ax.get_xaxis_transform(),
        fontsize=6.5,
        color="#555555",
    )
    ax.text(
        80,
        0.045,
        "Late / peak",
        ha="center",
        va="bottom",
        transform=ax.get_xaxis_transform(),
        fontsize=6.5,
        color="#555555",
    )

    for system_name in system_names:
        mean, sem = trajectories[system_name]
        color = SYSTEM_COLORS[system_name]
        ax.plot(phase, mean, color=color, linewidth=1.9, label=system_name)
        ax.fill_between(phase, mean - sem, mean + sem, color=color, alpha=0.16, linewidth=0)
        peak_i = int(np.argmax(mean))
        ax.plot(phase[peak_i], mean[peak_i], marker="o", color=color, markersize=3.2)

    ax.set_xlabel("Normalized exercise phase (%)")
    ax.set_ylabel("Mean normalized attribution")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", color="#D9D9D9", linestyle="--", linewidth=0.6, alpha=0.8)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.01, 0.99))

    out_base = args.out_dir / "Fig5c_system_phase_trajectory"
    save_all(fig, out_base, dpi=args.dpi)
    plt.close(fig)

    write_csv(
        args.out_dir / "Fig5c_system_phase_trajectory.csv",
        rows,
        [
            "phase_percent",
            "system",
            "mean_normalized_attribution",
            "sem_across_task_fold_curves",
            "n_task_fold_curves",
        ],
    )
    print(f"Saved Fig. 5c to {out_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
