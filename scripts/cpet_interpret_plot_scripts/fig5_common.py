# -*- coding: utf-8 -*-
"""Shared utilities for Fig. 5 clinical interpretability panels.

The plotting scripts live inside ``OurMethods/cpet_interpret_plot_scripts``.
By default they read sibling ``OurMethods/interpretation`` and write sibling
``OurMethods/interpret_auxB_figs``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib as mpl
import numpy as np


TASKS = ["t1", "t2", "t3", "t4", "t5"]

TASK_LABELS = {
    "t1": "t1 | CPET functional class",
    "t2": "t2 | Exercise capacity",
    "t3": "t3 | Exercise ECG interpretation",
    "t4": "t4 | Ventilatory function",
    "t5": "t5 | Heart-rate reserve",
}

TASK_LABELS_SHORT = {
    "t1": "t1\nCPET functional\nclass",
    "t2": "t2\nExercise\ncapacity",
    "t3": "t3\nExercise ECG\ninterpretation",
    "t4": "t4\nVentilatory\nfunction",
    "t5": "t5\nHeart-rate\nreserve",
}

SYSTEMS: List[Tuple[str, List[str]]] = [
    (
        "S0 Oxygen delivery",
        [
            "Load",
            "V'O2",
            "HR",
            "O2Pulse",
            "dO2/dW",
            "MET",
            "d(O2P)/dt",
            "VO2/kg",
            "dH/dO2",
            "SVc",
            "OUES",
        ],
    ),
    (
        "S1 Ventilatory drive",
        ["V'CO2", "V'E", "VTex", "BF", "RER", "BR"],
    ),
    (
        "S2 Gas-exchange efficiency",
        ["EqO2", "EqCO2", "PETO2", "PETCO2", "SpO2", "EqO2_COP", "VDc/VT"],
    ),
    (
        "S3 Reserve / stability",
        ["Psys", "Pdia", "PP", "HRR", "HR_diff", "d2(O2P)/dt2"],
    ),
]

SYSTEM_COLORS = {
    "S0 Oxygen delivery": "#2F6FA3",
    "S1 Ventilatory drive": "#C9792B",
    "S2 Gas-exchange efficiency": "#3B8C72",
    "S3 Reserve / stability": "#B24A5A",
}

SYSTEM_LABELS_SHORT = {
    "S0 Oxygen delivery": "Oxygen delivery",
    "S1 Ventilatory drive": "Ventilatory drive",
    "S2 Gas-exchange efficiency": "Gas-exchange efficiency",
    "S3 Reserve / stability": "Reserve / stability",
}

SYSTEM_LABELS_COMPACT = {
    "S0 Oxygen delivery": "Oxygen\ndelivery",
    "S1 Ventilatory drive": "Ventilatory\ndrive",
    "S2 Gas-exchange efficiency": "Gas-exchange\nefficiency",
    "S3 Reserve / stability": "Reserve /\nstability",
}

PHASE_BANDS = [
    (0, 20, "Early", "#F4F6F8"),
    (20, 60, "Mid", "#FFFFFF"),
    (60, 100, "Late / peak", "#F8F1E8"),
]


def default_root() -> Path:
    return Path(__file__).resolve().parent.parent / "interpretation"


def default_out_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "interpret_auxB_figs"


def set_pub_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def variable_to_system(variable_names: Sequence[str]) -> Dict[str, str]:
    idxs = system_indices(variable_names)
    return {
        variable_names[i]: system_name
        for system_name, system_idxs in idxs.items()
        for i in system_idxs
    }


def add_phase_bands(ax, label_y: float = 0.98, label_va: str = "top") -> None:
    for left, right, label, color in PHASE_BANDS:
        if color != "#FFFFFF":
            ax.axvspan(left, right, color=color, zorder=0)
        ax.text(
            (left + right) / 2,
            label_y,
            label,
            ha="center",
            va=label_va,
            transform=ax.get_xaxis_transform(),
            fontsize=6.7,
            color="#555555",
        )


def discover_folds(root: Path) -> List[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Interpretation root not found: {root}")
    folds = [
        p
        for p in sorted(root.iterdir(), key=lambda x: x.name)
        if p.is_dir() and (p / "variable_time_attr").exists()
    ]
    if not folds and (root / "variable_time_attr").exists():
        folds = [root]
    if not folds:
        raise FileNotFoundError(f"No fold folders with variable_time_attr under: {root}")
    return folds


def orient_time_variable(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D attribution array, got {a.shape}")
    if a.shape[1] == 30:
        return a
    if a.shape[0] == 30:
        return a.T
    raise ValueError(f"Cannot orient attribution array with shape {a.shape}")


def p99_normalize(a: np.ndarray) -> np.ndarray:
    scale = float(np.percentile(a, 99))
    if scale <= 0:
        return a.astype(float, copy=True)
    return np.clip(a / (scale + 1e-12), 0.0, 1.0)


def load_variable_names(folds: Sequence[Path]) -> List[str]:
    for fold in folds:
        mapping_file = fold / "variable_time_attr" / "channel_name_mapping.json"
        if not mapping_file.exists():
            continue
        obj = json.loads(mapping_file.read_text(encoding="utf-8"))
        mapping = obj.get("mapping", obj) if isinstance(obj, dict) else None
        if isinstance(mapping, dict):
            names = []
            for i in range(30):
                names.append(str(mapping.get(str(i), mapping.get(i, f"Var {i + 1:02d}"))))
            return names
    return [f"Var {i + 1:02d}" for i in range(30)]


def system_indices(variable_names: Sequence[str]) -> Dict[str, List[int]]:
    by_name = {name: i for i, name in enumerate(variable_names)}
    out: Dict[str, List[int]] = {}
    missing: List[str] = []
    for system_name, names in SYSTEMS:
        idxs = []
        for name in names:
            if name not in by_name:
                missing.append(name)
            else:
                idxs.append(by_name[name])
        out[system_name] = idxs
    if missing:
        raise ValueError(
            "Missing CPET variables required for system grouping: " + ", ".join(missing)
        )
    return out


def load_task_stack(
    folds: Sequence[Path],
    task: str,
    normalize_each_fold: bool = True,
) -> np.ndarray:
    arrs = []
    for fold in folds:
        path = fold / "variable_time_attr" / f"{task}_mean.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing attribution file: {path}")
        a = orient_time_variable(np.load(path))
        if normalize_each_fold:
            a = p99_normalize(a)
        arrs.append(a)
    return np.stack(arrs, axis=0)


def task_mean_matrix(
    folds: Sequence[Path],
    task: str,
    normalize_each_fold: bool = True,
) -> np.ndarray:
    return load_task_stack(folds, task, normalize_each_fold=normalize_each_fold).mean(axis=0)


def high_window(curve: np.ndarray, phase: np.ndarray, fraction: float = 0.8) -> Tuple[float, float, float]:
    curve = np.asarray(curve, dtype=float)
    peak_i = int(np.argmax(curve))
    peak_phase = float(phase[peak_i])
    threshold = float(curve[peak_i] * fraction)
    mask = curve >= threshold
    left = peak_i
    right = peak_i
    while left > 0 and mask[left - 1]:
        left -= 1
    while right < len(mask) - 1 and mask[right + 1]:
        right += 1
    return peak_phase, float(phase[left]), float(phase[right])


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_all(fig, out_base: Path, dpi: int = 600) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
