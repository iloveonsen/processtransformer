import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from processtransformer import constants
from processtransformer.data import loader
from processtransformer.models import transformer


def preprocess_single_trace(input_csv_path, output_csv_path, columns):
    """
    Preprocess a single trace CSV file.

    Args:
        input_csv_path: Path to input CSV file
        output_csv_path: Path to save processed CSV
        columns: List of column names [case_id, activity, timestamp]
    """
    # Read the CSV
    df = pd.read_csv(input_csv_path)

    # Rename columns to standard format
    df.columns = ["case:concept:name", "concept:name", "time:timestamp"]

    # Convert timestamp to datetime
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"],
                                          format="%Y-%m-%d %H:%M:%S")

    # Sort by timestamp
    df = df.sort_values("time:timestamp").reset_index(drop=True)

    # Calculate time features
    df["time_diff"] = df["time:timestamp"].diff().dt.total_seconds() / 86400  # in days
    df["time_diff"] = df["time_diff"].fillna(0)

    # Calculate cumulative time
    df["time_cumsum"] = df["time_diff"].cumsum()

    # Create prefix sequences
    processed_data = []
    activities = df["concept:name"].tolist()
    timestamps = df["time:timestamp"].tolist()

    for i in range(len(df)):
        prefix = " ".join(activities[:i+1])

        if i == 0:
            recent_time = 0
            latest_time = 0
        else:
            recent_time = (timestamps[i] - timestamps[i-1]).total_seconds() / 86400
            latest_time = (timestamps[i] - timestamps[0]).total_seconds() / 86400

        time_passed = df.loc[i, "time_cumsum"]

        processed_data.append({
            "case_id": df.loc[i, "case:concept:name"],
            "prefix": prefix,
            "k": i + 1,
            "recent_time": recent_time,
            "latest_time": latest_time,
            "time_passed": time_passed,
        })

    # Save processed data
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(output_csv_path, index=False)

    print(f"Processed data saved to: {output_csv_path}")
    return processed_df


def visualize_attention_scores(activity_scores, dataset_name, task_name, metric_value, output_path):
    """
    Visualize attention scores as a bar chart.

    Args:
        activity_scores: Dictionary mapping activity names to attention scores
        dataset_name: Name of the dataset
        task_name: Name of the task (e.g., "remaining_time")
        metric_value: Metric value to include in title (e.g., MAE)
        output_path: Path to save the visualization
    """
    # Sort activities by score for better visualization
    sorted_items = sorted(activity_scores.items(), key=lambda x: x[1], reverse=True)
    activities = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]

    # Create figure with high DPI for quality
    plt.figure(figsize=(12, 6), dpi=300)

    # Create bar chart with color mapping (darker = higher score)
    colors = plt.cm.Blues(np.array(scores) / max(scores) if max(scores) > 0 else np.zeros(len(scores)))
    bars = plt.bar(range(len(activities)), scores, color=colors)

    # Customize plot
    plt.xlabel('Activity', fontsize=12, fontweight='bold')
    plt.ylabel('Attention Score', fontsize=12, fontweight='bold')
    plt.title(f'{dataset_name}_{task_name}_{metric_value}', fontsize=14, fontweight='bold')
    plt.xticks(range(len(activities)), activities, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Apply tight layout for clean appearance
    plt.tight_layout()

    # Save with high quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Attention visualization saved to: {output_path}")


def aggregate_attention_scores(attn_weights, token_ids):
    """
    Aggregate attention scores across heads and positions.

    Args:
        attn_weights: Attention weights from transformer [batch, num_heads, seq_len, seq_len]
        token_ids: Token IDs of the input sequence [batch, seq_len]

    Returns:
        Dictionary mapping token_id to aggregated attention score
    """
    # attn_weights shape: [batch, num_heads, seq_len, seq_len]
    # We take the first (and only) batch
    attn = attn_weights[0]  # [num_heads, seq_len, seq_len]

    # Step 1: Element-wise sum across heads
    attn_summed = attn.sum(dim=0)  # [seq_len, seq_len]

    # Step 2: Row-wise sum to get per-token scores
    token_scores = attn_summed.sum(dim=0)  # [seq_len]

    # Step 3: Softmax normalization
    token_scores = F.softmax(token_scores, dim=0)

    # Map token IDs to scores
    token_ids_np = token_ids[0].cpu().numpy()  # [seq_len]
    token_scores_np = token_scores.cpu().numpy()  # [seq_len]

    # Aggregate scores for duplicate tokens
    score_dict = {}
    for token_id, score in zip(token_ids_np, token_scores_np):
        token_id = int(token_id)
        if token_id in score_dict:
            score_dict[token_id] += score
        else:
            score_dict[token_id] = score

    # Renormalize after aggregation
    total_score = sum(score_dict.values())
    score_dict = {k: v / total_score for k, v in score_dict.items()}

    return score_dict


def run_inference(args):
    """Main inference function."""

    # Construct paths based on dataset and inference directory
    dataset_inference_dir = os.path.join(args.inference_dir, args.dataset)
    raw_dir = os.path.join(dataset_inference_dir, "raw")
    processed_dir = os.path.join(dataset_inference_dir, "processed")
    results_dir = os.path.join(dataset_inference_dir, "results")

    # Input CSV path
    input_csv_path = os.path.join(raw_dir, args.input_csv)

    # Validate inputs
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv_path}\n"
            f"Please place your CSV file in: {raw_dir}/"
        )

    # Create output directories
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Step 1: Preprocess the single trace
    print("=" * 70)
    print("Step 1: Preprocessing single trace")
    print("=" * 70)

    input_filename = os.path.basename(args.input_csv)
    input_name = os.path.splitext(input_filename)[0]
    processed_csv_path = os.path.join(processed_dir, f"{input_name}_processed.csv")

    processed_df = preprocess_single_trace(
        input_csv_path,
        processed_csv_path,
        args.columns
    )

    # Step 2: Load metadata and model
    print("\n" + "=" * 70)
    print("Step 2: Loading metadata and model")
    print("=" * 70)

    metadata_path = os.path.join(args.data_dir, "processed", args.dataset, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    x_word_dict = metadata["x_word_dict"]
    y_word_dict = metadata["y_word_dict"]

    print(f"Loaded metadata from: {metadata_path}")
    print(f"  Vocabulary size: {len(x_word_dict)}")
    print(f"  Output classes: {len(y_word_dict)}")

    # Load model checkpoint
    model_path = os.path.join(args.model_dir, args.dataset, "remaining_time_ckpt.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=args.device)
    print(f"Loaded model from: {model_path}")

    # Get max case length from processed data
    max_case_length = max(len(prefix.split()) for prefix in processed_df["prefix"])

    # Create model
    model = transformer.get_remaining_time_model(
        max_case_length=max_case_length,
        vocab_size=len(x_word_dict),
        output_dim=1
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()

    print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")

    # Step 3: Prepare data for inference
    print("\n" + "=" * 70)
    print("Step 3: Preparing data for inference")
    print("=" * 70)

    # We'll use the last (longest) prefix for inference
    last_row = processed_df.iloc[-1]
    prefix = last_row["prefix"]
    time_features = np.array([[
        last_row["recent_time"],
        last_row["latest_time"],
        last_row["time_passed"]
    ]], dtype=np.float32)

    print(f"Input prefix: {prefix}")
    print(f"Time features: {time_features}")

    # Tokenize
    token_ids = [x_word_dict.get(token, 0) for token in prefix.split()]
    token_x = loader.pad_sequences([token_ids], maxlen=max_case_length)
    token_x = torch.tensor(token_x, dtype=torch.long).to(args.device)
    time_x = torch.tensor(time_features, dtype=torch.float32).to(args.device)

    # Step 4: Run inference
    print("\n" + "=" * 70)
    print("Step 4: Running inference")
    print("=" * 70)

    with torch.inference_mode():
        prediction, attn_weights = model(token_x, time_x, return_attention=True)

    prediction_value = prediction[0, 0].cpu().item()
    print(f"Predicted remaining time: {prediction_value:.4f} days")

    # Step 5: Aggregate attention scores
    print("\n" + "=" * 70)
    print("Step 5: Computing attention scores")
    print("=" * 70)

    attention_scores = aggregate_attention_scores(attn_weights, token_x)

    # Map token IDs back to activities
    id_to_activity = {int(v): k for k, v in x_word_dict.items()}
    activity_scores = {}
    for token_id, score in attention_scores.items():
        if token_id in id_to_activity:
            activity = id_to_activity[token_id]
            activity_scores[activity] = score
            print(f"  {activity}: {score:.6f}")

    # Step 6: Create output CSV
    print("\n" + "=" * 70)
    print("Step 6: Saving results")
    print("=" * 70)

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    output_filename = f"{input_name}_{timestamp}.csv"
    output_path = os.path.join(results_dir, output_filename)

    # Create output row
    output_row = {"prediction": prediction_value}

    # Add all possible activities with scores (0 if not present)
    for activity in y_word_dict.keys():
        output_row[activity] = activity_scores.get(activity, 0.0)

    # Save to CSV
    output_df = pd.DataFrame([output_row])
    output_df.to_csv(output_path, index=False)

    print(f"Results saved to: {output_path}")

    # Step 7: Visualize attention scores
    print("\n" + "=" * 70)
    print("Step 7: Visualizing attention scores")
    print("=" * 70)

    # Get metric value from checkpoint (MAE for remaining_time)
    metric_value = checkpoint.get("best_mae", "unknown")
    if isinstance(metric_value, float):
        metric_value = f"mae_{metric_value:.4f}"
    else:
        metric_value = "mae_unknown"

    # Create visualization path (same name as CSV but with .png extension)
    viz_filename = f"{input_name}_{timestamp}.png"
    viz_path = os.path.join(results_dir, viz_filename)

    # Generate visualization
    visualize_attention_scores(
        activity_scores=activity_scores,
        dataset_name=args.dataset,
        task_name="remaining_time",
        metric_value=metric_value,
        output_path=viz_path
    )

    print("\n" + "=" * 70)
    print("Inference completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Transformer - Remaining Time Inference")

    parser.add_argument("--dataset", required=True, type=str,
                        help="dataset name")

    parser.add_argument("--input_csv", required=True, type=str,
                        help="input CSV filename (e.g., test_trace.csv). "
                             "File should be placed in {inference_dir}/{dataset}/raw/")

    parser.add_argument("--data_dir", default="./datasets", type=str,
                        help="data directory containing processed data and metadata")

    parser.add_argument("--model_dir", default="./models", type=str,
                        help="model directory containing trained models")

    parser.add_argument("--inference_dir", default="./inference", type=str,
                        help="inference directory (structure: {inference_dir}/{dataset}/raw|processed|results)")

    parser.add_argument("--columns", type=str, nargs='+',
                        default=["case:concept:name", "concept:name", "time:timestamp"],
                        help="column names in input CSV")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device to run inference on")

    args = parser.parse_args()

    run_inference(args)
