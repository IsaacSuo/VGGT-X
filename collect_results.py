#!/usr/bin/env python3
"""
Collect and compare evaluation results from VGGT and DA3 experiments.
Usage: python collect_results.py /path/to/mipnerf360
"""

import os
import sys
import re
from pathlib import Path

SCENES = ["garden", "counter", "room", "bonsai", "bicycle", "stump", "kitchen"]
METHODS = [
    ("VGGT", "_vggt"),
    ("VGGT+GA", "_vggt_ga"),
    ("DA3", "_da3"),
    ("DA3+GA", "_da3_ga"),
]


def parse_results_file(filepath):
    """Parse results.txt and extract metrics."""
    if not os.path.exists(filepath):
        return None

    metrics = {}
    with open(filepath, 'r') as f:
        content = f.read()

    # Parse AUC metrics
    auc_match = re.search(r'AUC@(\d+):\s*([\d.]+)', content)
    if auc_match:
        metrics['auc'] = float(auc_match.group(2))

    # Parse pose error
    pose_match = re.search(r'Pose Error.*?:\s*([\d.]+)', content, re.IGNORECASE)
    if pose_match:
        metrics['pose_err'] = float(pose_match.group(1))

    # Parse point cloud metrics
    acc_match = re.search(r'Acc(?:uracy)?.*?:\s*([\d.]+)', content, re.IGNORECASE)
    if acc_match:
        metrics['accuracy'] = float(acc_match.group(1))

    comp_match = re.search(r'Comp(?:leteness)?.*?:\s*([\d.]+)', content, re.IGNORECASE)
    if comp_match:
        metrics['completeness'] = float(comp_match.group(1))

    # Parse time and memory
    time_match = re.search(r'Time.*?:\s*([\d.]+)', content, re.IGNORECASE)
    if time_match:
        metrics['time'] = float(time_match.group(1))

    mem_match = re.search(r'Memory.*?:\s*([\d.]+)', content, re.IGNORECASE)
    if mem_match:
        metrics['memory'] = float(mem_match.group(1))

    return metrics if metrics else None


def main(dataset_root):
    dataset_root = Path(dataset_root)
    parent_dir = dataset_root.parent

    print(f"Dataset root: {dataset_root}")
    print(f"Looking for results in: {parent_dir}")
    print()

    # Collect all results
    results = {}
    for scene in SCENES:
        results[scene] = {}
        for method_name, suffix in METHODS:
            result_dir = parent_dir / f"{dataset_root.name}{suffix}" / scene
            result_file = result_dir / "results.txt"
            metrics = parse_results_file(result_file)
            results[scene][method_name] = metrics

    # Print comparison table
    print("=" * 100)
    print("COMPARISON RESULTS")
    print("=" * 100)

    # Header
    print(f"\n{'Scene':<12}", end="")
    for method_name, _ in METHODS:
        print(f" | {method_name:^18}", end="")
    print()
    print("-" * 12 + ("|" + "-" * 20) * len(METHODS))

    # Per-scene results
    for scene in SCENES:
        print(f"{scene:<12}", end="")
        for method_name, _ in METHODS:
            metrics = results[scene].get(method_name)
            if metrics:
                # Show AUC or accuracy if available
                if 'auc' in metrics:
                    val = f"AUC: {metrics['auc']:.3f}"
                elif 'accuracy' in metrics:
                    val = f"Acc: {metrics['accuracy']:.3f}"
                else:
                    val = "Partial"
            else:
                val = "N/A"
            print(f" | {val:^18}", end="")
        print()

    # Summary statistics
    print()
    print("=" * 100)
    print("AVERAGE METRICS")
    print("=" * 100)

    for method_name, _ in METHODS:
        aucs = []
        accs = []
        times = []
        for scene in SCENES:
            metrics = results[scene].get(method_name)
            if metrics:
                if 'auc' in metrics:
                    aucs.append(metrics['auc'])
                if 'accuracy' in metrics:
                    accs.append(metrics['accuracy'])
                if 'time' in metrics:
                    times.append(metrics['time'])

        print(f"\n{method_name}:")
        if aucs:
            print(f"  Avg AUC: {sum(aucs)/len(aucs):.4f} (over {len(aucs)} scenes)")
        if accs:
            print(f"  Avg Accuracy: {sum(accs)/len(accs):.4f} (over {len(accs)} scenes)")
        if times:
            print(f"  Avg Time: {sum(times)/len(times):.2f}s (over {len(times)} scenes)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python collect_results.py /path/to/mipnerf360")
        sys.exit(1)

    main(sys.argv[1])
