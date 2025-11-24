#!/usr/bin/env python3
"""
Visualize integrated batch inference results for selected case IDs.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def visualize_attention_scores(activity_scores, case_id, output_path):
    """
    Visualize attention scores as a bar chart.

    Args:
        activity_scores: Dictionary mapping activity names to attention scores
        case_id: Case ID for the title
        output_path: Path to save the visualization
    """
    # NOT Sort activities by score for better visualization
    items = list(activity_scores.items())
    activities = [item[0] for item in items]
    scores = [item[1] for item in items]

    # Create figure with high DPI for quality
    plt.figure(figsize=(12, 6), dpi=300)

    # Create bar chart with color mapping (darker = higher score)
    colors = plt.cm.Blues(np.array(scores) / max(scores) if max(scores) > 0 else np.zeros(len(scores)))
    bars = plt.bar(range(len(activities)), scores, color=colors)

    # Add value labels on top of each bar
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=8)

    # Customize plot
    plt.xlabel('Activity', fontsize=12, fontweight='bold')
    plt.ylabel('Attention Score', fontsize=12, fontweight='bold')
    plt.title(f'Integrated Attention Scores - {case_id}', fontsize=14, fontweight='bold')
    plt.xticks(range(len(activities)), activities, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Apply tight layout for clean appearance
    plt.tight_layout()

    # Save with high quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def visualize_integrated_results(integrated_json_path, case_ids, output_dir):
    """
    Visualize integrated attention scores for selected case IDs.

    Args:
        integrated_json_path: Path to integrated.json file
        case_ids: List of case IDs to visualize
        output_dir: Directory to save visualization plots
    """
    # Load integrated results
    print(f"Loading integrated results from: {integrated_json_path}")
    with open(integrated_json_path, 'r', encoding='utf-8') as f:
        integrated = json.load(f)

    print(f"Total cases in integrated file: {len(integrated)}")
    print(f"Cases to visualize: {len(case_ids)}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Visualize each case
    print("\nGenerating visualizations...")
    missing_cases = []

    for i, case_id in enumerate(case_ids, 1):
        if case_id not in integrated:
            missing_cases.append(case_id)
            print(f"  [{i}/{len(case_ids)}] ✗ Case '{case_id}' not found in integrated results")
            continue

        # Get attention scores for this case
        activity_scores = integrated[case_id]

        # Generate output filename with case ID
        output_file = output_path / f"attention_scores_{case_id}.png"

        # Create visualization
        print(f"  [{i}/{len(case_ids)}] Visualizing case: {case_id}")
        visualize_attention_scores(activity_scores, case_id, output_file)

    # Summary
    print(f"\n{'='*60}")
    print(f"Visualization complete!")
    print(f"  Successfully visualized: {len(case_ids) - len(missing_cases)} cases")
    if missing_cases:
        print(f"  Missing cases: {len(missing_cases)}")
        print(f"    {missing_cases}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Path to integrated.json file
    integrated_json_path = './output/integrated/integrated.json'

    # List of case IDs to visualize
    case_ids = [
        'Application_1000386745',
        'Application_1000474975',
        'Application_1000557783',
        # Add more case IDs as needed
    ]

    # Output directory for visualizations
    output_dir = './viz/output'

    # Generate visualizations
    visualize_integrated_results(integrated_json_path, case_ids, output_dir)
