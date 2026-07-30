"""
Gate Usage Analysis Tool
========================

Analyzes expert selection patterns from trained ProtectedDualEngineMTL models.

Usage:
    python tools/summarize_gate_usage.py --checkpoint models/mtl_best.pth

Output:
    - Expert usage distribution per task
    - Gate weight evolution across epochs (if history available)
    - Task-expert correlation heatmap
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict


def load_gate_weights(checkpoint_path):
    """
    Load gate weights from checkpoint

    Args:
        checkpoint_path: Path to .pth file

    Returns:
        gate_weights: Dict mapping task_key to expert weights
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    gate_weights = {}

    # Extract gate projection weights
    for key, value in state_dict.items():
        if "alpha_gates" in key and "gate_proj.weight" in key:
            task_key = key.split(".")[1]  # e.g., "alpha_gates.t1.gate_proj.weight" -> "t1"
            # gate_proj.weight: [num_experts, context_dim]
            # We want the bias or output after softmax, but weight gives us learned patterns
            gate_weights[task_key] = value.numpy()
        elif "beta_gates" in key and "gate_proj.weight" in key:
            task_key = key.split(".")[1]
            gate_weights[task_key] = value.numpy()

    return gate_weights


def analyze_gate_patterns(gate_weights):
    """
    Analyze gate weight patterns

    Args:
        gate_weights: Dict mapping task_key to weight arrays

    Returns:
        analysis: Dict with usage statistics
    """
    analysis = {}

    for task_key, weights in gate_weights.items():
        # weights shape: [num_experts, context_dim]
        # Compute average magnitude per expert (row-wise)
        expert_magnitudes = np.abs(weights).sum(axis=1)

        # Normalize to get relative importance
        expert_importance = expert_magnitudes / expert_magnitudes.sum()

        analysis[task_key] = {
            "expert_importance": expert_importance,
            "dominant_expert": np.argmax(expert_importance),
            "dominance_ratio": expert_importance.max() / expert_importance.mean(),
            "diversity": 1.0 - (expert_importance.max() - expert_importance.min()),
        }

    return analysis


def plot_gate_heatmap(analysis, save_path):
    """
    Plot expert usage heatmap

    Args:
        analysis: Analysis results dict
        save_path: Path to save figure
    """
    # Collect all tasks and experts
    tasks = sorted(analysis.keys())

    # Determine max number of experts
    max_experts = max(len(analysis[t]["expert_importance"]) for t in tasks)

    # Build matrix
    matrix = np.zeros((len(tasks), max_experts))

    for i, task_key in enumerate(tasks):
        importance = analysis[task_key]["expert_importance"]
        matrix[i, :len(importance)] = importance

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels(tasks)

    ax.set_xticks(np.arange(max_experts))
    ax.set_xticklabels([f"Expert {i}" for i in range(max_experts)])

    ax.set_xlabel("Expert Index")
    ax.set_ylabel("Task")

    plt.colorbar(im, ax=ax, label="Relative Importance")

    plt.title("Expert Usage Distribution Across Tasks")
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()

    print(f"Saved heatmap to {save_path}")


def print_summary(analysis):
    """
    Print analysis summary to console
    """
    print("\n" + "=" * 60)
    print("Gate Usage Analysis Summary")
    print("=" * 60)

    for task_key, stats in sorted(analysis.items()):
        print(f"\n{task_key}:")
        print(f"  Expert importance: {stats['expert_importance']}")
        print(f"  Dominant expert: {stats['dominant_expert']}")
        print(f"  Dominance ratio: {stats['dominance_ratio']:.2f}")
        print(f"  Diversity: {stats['diversity']:.2f}")

    # Overall statistics
    avg_dominance = np.mean([s["dominance_ratio"] for s in analysis.values()])
    avg_diversity = np.mean([s["diversity"] for s in analysis.values()])

    print("\n" + "-" * 40)
    print(f"Average dominance ratio: {avg_dominance:.2f}")
    print(f"Average diversity: {avg_diversity:.2f}")

    if avg_dominance > 2.0:
        print("\n[Warning] High dominance ratio suggests gate collapse!")
        print("  Consider: Increase entropy regularization")
    elif avg_diversity < 0.3:
        print("\n[Warning] Low diversity suggests uneven expert usage")
        print("  Consider: Adjust temperature annealing schedule")
    else:
        print("\n[Good] Expert selection is reasonably diverse")


def main():
    parser = argparse.ArgumentParser(description="Analyze gate usage patterns")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="gate_analysis", help="Output directory")

    args = parser.parse_args()

    # Load gate weights
    print(f"Loading checkpoint: {args.checkpoint}")
    gate_weights = load_gate_weights(args.checkpoint)

    if not gate_weights:
        print("[Error] No gate weights found in checkpoint")
        print("  Check if the checkpoint is from ProtectedDualEngineMTL")
        return

    # Analyze
    analysis = analyze_gate_patterns(gate_weights)

    # Print summary
    print_summary(analysis)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot heatmap
    plot_heatmap_path = output_dir / "gate_heatmap.png"
    plot_gate_heatmap(analysis, str(plot_heatmap_path))

    # Save analysis JSON
    json_path = output_dir / "analysis.json"
    with open(json_path, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_analysis = {}
        for task_key, stats in analysis.items():
            json_analysis[task_key] = {
                "expert_importance": stats["expert_importance"].tolist(),
                "dominant_expert": int(stats["dominant_expert"]),
                "dominance_ratio": float(stats["dominance_ratio"]),
                "diversity": float(stats["diversity"]),
            }
        json.dump(json_analysis, f, indent=2)

    print(f"Saved analysis to {json_path}")


if __name__ == "__main__":
    main()
