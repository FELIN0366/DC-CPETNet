#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot development-vs-holdout functional label balance for CPET MTL Figure 1C.

Input:
  Excel workbook with sheet `Functional_Label_Distribution` containing columns like:
    Task | Label | Overall (N=1136) | Development (n=909) | Validation (n=227)

Output:
  A wide multi-panel 100% stacked bar figure for t1–t5 label distributions
  in the development set and independent holdout test set.

Key plotting rules in this version:
  1) No figure title/subtitle is drawn.
  2) Larger label proportions are stacked at the bottom; smaller proportions at the top.
  3) Each segment is annotated with both count and percentage.
  4) Legends are placed horizontally under each panel.
  5) Colors are drawn from the provided model-palette color set.

Example:
  python plot_figure1c_label_balance_revised.py \
      --excel cpet_features_statistics_by_dev_validation_20260706_151013.xlsx \
      --outdir ./figures \
      --prefix Figure1C_label_balance

Notes:
  - The script treats the column containing "Validation" or "Holdout" as the
    independent holdout test set for display, because the provided workbook
    stores the holdout column as `Validation (n=227)`.
  - Vector outputs are saved with editable text where possible.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


TASK_ORDER = ["t1", "t2", "t3", "t4", "t5"]

TASK_TITLES = {
    "t1": "t1  Weber functional class",
    "t2": "t2  Exercise capacity",
    "t3": "t3  Exercise ECG response",
    "t4": "t4  Breathing reserve",
    "t5": "t5  Heart-rate reserve",
}

LABEL_MAP = {
    # t1
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
    # t2
    "正常及大致正常": "Normal / near-normal",
    "正常/大致正常": "Normal / near-normal",
    "中度及轻度下降": "Mild–moderate reduction",
    "轻度及中度下降": "Mild–moderate reduction",
    "重度及极重度下降": "Severe–extreme reduction",
    "重度/极重度下降": "Severe–extreme reduction",
    # t3
    "阳": "Positive",
    "阴": "Negative",
    # t4
    "下降": "Impaired",
    "正常": "Normal",
    # t5
    "未用尽": "Not exhausted",
    "用尽": "Exhausted",
}

# User-provided palette. Values are reused for label categories.

# Task-specific semantic mapping based on the provided palette.
# Good/normal or majority non-event labels use blue/green; impaired/adverse labels use red/orange.
TASK_COLORS = {
    "t1": {
        "A":  "#1F77B4",
        "B":  "#2CA02C",
        "C":  "#FF7F0E",
        "D":  "#D62728",
    },
    "t2": {
        "Normal / near-normal":   "#2CA02C",
        "Mild–moderate reduction":"#1F77B4",
        "Severe–extreme reduction": "#FF7F0E",
    },
    "t3": {
        "Negative":  "#1F77B4",
        "Positive": "#FF7F0E",
    },
    "t4": {
        "Normal":  "#1F77B4",
        "Impaired": "#FF7F0E",
    },
    "t5": {
        "Not exhausted":  "#1F77B4",
        "Exhausted":  "#FF7F0E",
    },
}


def parse_count_percent(value: object) -> Tuple[int, float]:
    """Parse cells such as '394 (43.3%)' into (394, 43.3)."""
    if pd.isna(value):
        return 0, 0.0
    text = str(value).strip()
    match = re.search(r"([0-9,]+)\s*\(([-+]?\d*\.?\d+)\s*%\)", text)
    if not match:
        raise ValueError(f"Cannot parse count/percent cell: {value!r}")
    count = int(match.group(1).replace(",", ""))
    percent = float(match.group(2))
    return count, percent


def extract_task_id(task_name: str) -> str:
    """Extract t1–t5 from the task column."""
    match = re.match(r"\s*(t[1-5])\b", str(task_name), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot extract task id from task value: {task_name!r}")
    return match.group(1).lower()


def extract_n_from_colname(colname: str) -> int | None:
    """Extract n from a column name like 'Development (n=909)' or 'Overall (N=1136)'."""
    match = re.search(r"[Nn]\s*=\s*([0-9,]+)", colname)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def choose_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Identify development and holdout columns from the workbook."""
    dev_candidates = [c for c in df.columns if "Development" in str(c)]
    hold_candidates = [
        c
        for c in df.columns
        if ("Holdout" in str(c)) or ("Validation" in str(c)) or ("Independent" in str(c))
    ]
    if not dev_candidates:
        raise ValueError("No development column found. Expected a column containing 'Development'.")
    if not hold_candidates:
        raise ValueError(
            "No holdout/validation column found. Expected a column containing 'Holdout', 'Validation', or 'Independent'."
        )
    return dev_candidates[0], hold_candidates[0]


def load_label_distribution(excel_path: Path, sheet_name: str) -> Tuple[pd.DataFrame, Dict[str, int | None]]:
    """Load and normalize the functional label distribution sheet."""
    raw = pd.read_excel(excel_path, sheet_name=sheet_name)
    required = {"Task", "Label"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns in sheet {sheet_name!r}: {sorted(missing)}")

    dev_col, hold_col = choose_columns(raw)
    meta = {
        "development_n": extract_n_from_colname(dev_col),
        "holdout_n": extract_n_from_colname(hold_col),
    }

    rows: List[dict] = []
    for _, row in raw.iterrows():
        task_id = extract_task_id(row["Task"])
        original_label = str(row["Label"]).strip()
        label = LABEL_MAP.get(original_label, original_label)
        for split_key, split_name, col in [
            ("development", "Development", dev_col),
            ("holdout", "Holdout", hold_col),
        ]:
            n, pct = parse_count_percent(row[col])
            rows.append(
                {
                    "task": task_id,
                    "task_name": TASK_TITLES.get(task_id, str(row["Task"])),
                    "label_original": original_label,
                    "label": label,
                    "split": split_key,
                    "split_name": split_name,
                    "n": n,
                    "percent": pct,
                }
            )
    data = pd.DataFrame(rows)

    # Preserve workbook label order for fallback, but actual stack order is determined later
    # by average split proportion so that larger proportions are plotted at the bottom.
    label_order = (
        data[["task", "label"]]
        .drop_duplicates()
        .assign(task_order=lambda x: x["task"].map({t: i for i, t in enumerate(TASK_ORDER)}))
        .sort_values(["task_order"])
    )
    label_rank = {(r.task, r.label): i for i, r in enumerate(label_order.itertuples(index=False))}
    data["task_order"] = data["task"].map({t: i for i, t in enumerate(TASK_ORDER)})
    data["label_order_original"] = data.apply(lambda r: label_rank[(r["task"], r["label"])], axis=1)
    data["split_order"] = data["split"].map({"development": 0, "holdout": 1})
    data = data.sort_values(["task_order", "label_order_original", "split_order"]).reset_index(drop=True)
    return data, meta


def _wrap_label(label: str) -> str:
    """Shorten/wrap label text for compact horizontal legends."""
    replacements = {
        "Normal / near-normal": "Normal/near-normal",
        "Mild–moderate reduction": "Mild–moderate reduction",
        "Severe–extreme reduction": "Severe–extreme reduction",
        "Not exhausted": "Not exhausted",
    }
    return replacements.get(label, label)


def choose_available_font(candidates: Iterable[str] = ("Arial", "Liberation Sans", "DejaVu Sans")) -> str:
    """Return the first available font family from a candidate list."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in candidates:
        if font in available:
            return font
    return "DejaVu Sans"


def _hex_to_rgb01(hex_color: str) -> Tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _text_color_for_fill(hex_color: str) -> str:
    """Choose black/white text for contrast against a fill color."""
    r, g, b = _hex_to_rgb01(hex_color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.58 else "white"


def _labels_sorted_largest_bottom(task_df: pd.DataFrame) -> List[str]:
    """Return labels ordered from largest average proportion to smallest.

    Matplotlib stacks in iteration order, so this makes the largest label start
    from the bottom of the stacked bar.
    """
    label_stats = (
        task_df.groupby("label", as_index=False)
        .agg(mean_percent=("percent", "mean"), original_order=("label_order_original", "min"))
        .sort_values(["mean_percent", "original_order"], ascending=[False, True])
    )
    return label_stats["label"].tolist()


def plot_label_balance(
    data: pd.DataFrame,
    meta: Dict[str, int | None],
    output_prefix: Path,
    dpi: int = 600,
    figure_width: float = 18.0,
    figure_height: float = 4.6,
) -> None:
    """Create a wide 5-panel development-vs-holdout stacked bar figure."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
        }
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=5,
        figsize=(figure_width, figure_height),
        sharey=True,
        constrained_layout=False,
    )

    split_display = {
        "development": f"Development\n(n={meta.get('development_n') or ''})".rstrip(),
        "holdout": f"Holdout\n(n={meta.get('holdout_n') or ''})".rstrip(),
    }

    for ax, task in zip(axes, TASK_ORDER):
        task_df = data[data["task"] == task].copy()
        if task_df.empty:
            ax.axis("off")
            continue

        x = np.array([0, 1], dtype=float)
        bottoms = np.zeros(2)
        labels = _labels_sorted_largest_bottom(task_df)

        for label in labels:
            label_df = task_df[task_df["label"] == label].sort_values("split_order")
            heights = label_df["percent"].to_numpy(dtype=float)
            counts = label_df["n"].to_numpy(dtype=int)
            color = TASK_COLORS.get(task, {}).get(label, "#8C8C8C")
            bars = ax.bar(
                x,
                heights,
                bottom=bottoms,
                width=0.70,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                label=_wrap_label(label),
            )

            # In-segment annotation: count + percentage.
            # For very small segments, keep the label concise to avoid clutter.
            for i, (bar, h, n) in enumerate(zip(bars, heights, counts)):
                if h >= 7:
                    annotation = f"{n}\n({h:.1f}%)"
                    if h >= 7 and h < 17:
                        annotation = f"{n} ({h:.1f}%)"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottoms[i] + h / 2,
                        annotation,
                        ha="center",
                        va="center",
                        fontsize=10,
                        linespacing=0.9,
                        fontweight="bold",
                        color=_text_color_for_fill(color),
                    )
                elif h >= 3:
                    annotation = f"{n} ({h:.1f}%)"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottoms[i] + h + 1.2,
                        annotation,
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                        color="black",
                    )
            bottoms += heights

        ax.set_title(TASK_TITLES.get(task, task), pad=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([split_display["development"], split_display["holdout"]])
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("Label proportion (%)")
        else:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)

        # Horizontal legend for every panel.
        legend_ncol = 2 if task == "t2" else max(1, len(labels))

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            frameon=False,
            ncol=legend_ncol,
            handlelength=1.0,
            handletextpad=0.35,
            columnspacing=0.75,
            borderaxespad=0,
        )

    # No global title/subtitle. Keep the subplot geometry close to the previous version.
    fig.subplots_adjust(left=0.045, right=0.998, top=0.82, bottom=0.30, wspace=0.12)

    for ext in ["png", "pdf", "svg"]:
        out = output_prefix.with_suffix(f".{ext}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def export_long_table(data: pd.DataFrame, output_path: Path) -> None:
    """Save the normalized long table used for plotting."""
    cols = ["task", "task_name", "split", "split_name", "label_original", "label", "n", "percent"]
    data[cols].to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot t1–t5 development-vs-holdout label balance.")
    parser.add_argument(
        "--excel",
        type=Path,
        default=Path("xx_path"),
        help="Input Excel workbook.",
    )
    parser.add_argument(
        "--sheet",
        default="Functional_Label_Distribution",
        help="Sheet name containing label distributions.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("xx_path"), help="Output directory.")
    parser.add_argument(
        "--prefix",
        default="Figure1C_functional_label_balance_large",
        help="Output filename prefix without extension.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution.")
    parser.add_argument("--width", type=float, default=18.0, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=4.6, help="Figure height in inches.")
    args = parser.parse_args()

    if not args.excel.exists():
        raise FileNotFoundError(f"Excel file not found: {args.excel}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    data, meta = load_label_distribution(args.excel, args.sheet)
    export_long_table(data, args.outdir / f"{args.prefix}_long.csv")
    plot_label_balance(
        data=data,
        meta=meta,
        output_prefix=args.outdir / args.prefix,
        dpi=args.dpi,
        figure_width=args.width,
        figure_height=args.height,
    )
    print(f"Saved figure files to: {args.outdir.resolve()}")
    print(f"Saved normalized long table: {(args.outdir / f'{args.prefix}_long.csv').resolve()}")


if __name__ == "__main__":
    main()

