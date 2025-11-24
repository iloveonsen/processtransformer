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
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

from processtransformer import constants
from processtransformer.data import loader
from processtransformer.models import transformer


def preprocess_multiple_traces(input_csv_path, output_csv_path, columns):
    """
    Preprocess multiple traces from a CSV file.
    Each case is processed separately but saved together.

    Args:
        input_csv_path: Path to input CSV file (can contain multiple cases)
        output_csv_path: Path to save processed CSV
        columns: List of column names [case_id, activity, timestamp]

    Returns:
        processed_df: DataFrame with all processed cases
        case_ids: List of unique case IDs
    """
    # Read the CSV
    df = pd.read_csv(input_csv_path)

    # Rename columns to standard format
    df.columns = ["case:concept:name", "concept:name", "time:timestamp"]

    df["concept:name"] = df["concept:name"].str.lower()
    df["concept:name"] = df["concept:name"].str.replace(" ", "-")
    df["time:timestamp"] = df["time:timestamp"].str.replace("/", "-")
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], format="%Y-%m-%d %H:%M:%S")

    # Sort by case and timestamp
    df = df.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)

    # Get unique case IDs
    case_ids = df["case:concept:name"].unique().tolist()

    print(f"Found {len(case_ids)} unique cases in the input file")

    # Process each case using groupby (much faster than repeated filtering)
    all_processed_data = []

    for case_id, case_df in tqdm(df.groupby("case:concept:name", sort=False),
                                  desc="Preprocessing cases",
                                  total=len(case_ids)):
        case_df = case_df.reset_index(drop=True)

        # Extract activities and timestamps as lists
        activities = case_df["concept:name"].tolist()
        timestamps = case_df["time:timestamp"].tolist()

        # Calculate time differences
        time_diffs = [0.0]  # First event has 0 time diff
        for i in range(1, len(timestamps)):
            time_diffs.append((timestamps[i] - timestamps[i-1]).total_seconds() / 86400)

        # Cumulative sum
        time_cumsum = 0
        prefix = activities[0]  # Initialize with first activity

        for i in range(len(case_df)):
            if i == 0:
                recent_time = 0
                latest_time = 0
            elif i == 1:
                recent_time = time_diffs[i]
                latest_time = time_diffs[i]
            else:
                recent_time = time_diffs[i]
                latest_time = (timestamps[i] - timestamps[0]).total_seconds() / 86400

            time_cumsum += time_diffs[i]

            all_processed_data.append({
                "case_id": case_id,
                "prefix": prefix,
                "k": i + 1,
                "recent_time": recent_time,
                "latest_time": latest_time,
                "time_passed": time_cumsum,
            })

            # Incrementally build prefix for next iteration
            if i + 1 < len(activities):
                prefix = prefix + " " + activities[i + 1]

    # Save processed data
    processed_df = pd.DataFrame(all_processed_data)
    processed_df.to_csv(output_csv_path, index=False)

    print(f"Processed data saved to: {output_csv_path}")
    print(f"  Total prefixes: {len(processed_df)}")
    print(f"  Cases: {len(case_ids)}")

    return processed_df, case_ids


class InferenceDataset(Dataset):
    """Dataset for batch inference."""

    def __init__(self, token_x, time_x, case_ids):
        """
        Args:
            token_x: Tensor of token IDs [num_samples, max_case_length]
            time_x: Tensor of time features [num_samples, 3]
            case_ids: List of case IDs
        """
        self.token_x = token_x
        self.time_x = time_x
        self.case_ids = case_ids

    def __len__(self):
        return len(self.token_x)

    def __getitem__(self, idx):
        return {
            'token_x': self.token_x[idx],
            'time_x': self.time_x[idx],
            'case_id': self.case_ids[idx]
        }


def aggregate_attention_scores_batch(attn_weights, token_ids):
    """
    Aggregate attention scores across heads and positions for a batch.

    Args:
        attn_weights: Attention weights [batch, num_heads, seq_len, seq_len]
        token_ids: Token IDs [batch, seq_len]

    Returns:
        List of dictionaries mapping token_id to aggregated attention score
    """
    batch_size = attn_weights.shape[0]
    results = []

    for b in range(batch_size):
        # attn shape for this sample: [num_heads, seq_len, seq_len]
        attn = attn_weights[b]

        # Step 1: Sum across heads
        attn_summed = attn.sum(dim=0)  # [seq_len, seq_len]

        # Step 2: Sum across source positions to get total attention received
        token_scores = attn_summed.sum(dim=0)  # [seq_len]

        # Step 3: Softmax normalization
        token_scores = F.softmax(token_scores, dim=0)

        # Map token IDs to scores
        token_ids_np = token_ids[b].cpu().numpy()
        token_scores_np = token_scores.cpu().numpy()

        # Aggregate scores for duplicate tokens (excluding [PAD] and [UNK])
        score_dict = {}
        for token_id, score in zip(token_ids_np, token_scores_np):
            token_id = int(token_id)
            # Skip special tokens: [PAD]=0, [UNK]=1
            if token_id == 0 or token_id == 1:
                continue
            if token_id in score_dict:
                score_dict[token_id] += score
            else:
                score_dict[token_id] = score

        # Renormalize after excluding special tokens
        total_score = sum(score_dict.values())
        if total_score > 0:
            score_dict = {k: v / total_score for k, v in score_dict.items()}

        results.append(score_dict)

    return results


def run_batch_inference(args):
    """Main batch inference function."""

    # Construct paths
    dataset_inference_dir = os.path.join(args.inference_dir, args.dataset)
    raw_dir = os.path.join(dataset_inference_dir, "raw")
    processed_dir = os.path.join(dataset_inference_dir, "processed")
    results_dir = os.path.join(dataset_inference_dir, "results")

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

    # Step 1: Preprocess multiple traces
    print("=" * 70)
    print("Step 1: Preprocessing multiple traces")
    print("=" * 70)

    input_filename = os.path.basename(args.input_csv)
    input_name = os.path.splitext(input_filename)[0]
    processed_csv_path = os.path.join(processed_dir, f"{input_name}_processed.csv")

    processed_df, case_ids = preprocess_multiple_traces(
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

    checkpoint = torch.load(model_path, map_location=args.device, weights_only=False)
    print(f"Loaded model from: {model_path}")

    # Get max case length from checkpoint
    max_case_length = checkpoint["model_state_dict"]["embedding.pos_emb.weight"].shape[0]
    print(f"  Max case length (from training): {max_case_length}")

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

    # Load y_scaler if available
    y_scaler = None
    if "y_scaler" in checkpoint:
        import pickle
        y_scaler = pickle.loads(checkpoint["y_scaler"])
        print("✓ y_scaler loaded (predictions will be inverse-transformed)")
    else:
        print("⚠️  WARNING: y_scaler not found. Predictions will be scaled values.")

    # Step 3: Prepare data for batch inference
    print("\n" + "=" * 70)
    print("Step 3: Preparing data for batch inference")
    print("=" * 70)

    # Get the last (longest) prefix for each case (using groupby for efficiency)
    last_prefixes = []
    batch_case_ids = []

    # Group by case_id and get last row of each group
    for case_id, group in tqdm(processed_df.groupby("case_id", sort=False),
                                desc="Preparing cases",
                                total=len(case_ids)):
        last_row = group.iloc[-1]

        prefix = last_row["prefix"]
        time_features = np.array([
            last_row["recent_time"],
            last_row["latest_time"],
            last_row["time_passed"]
        ], dtype=np.float32)

        # Tokenize
        token_ids = [x_word_dict.get(token, x_word_dict.get("[UNK]", 0)) for token in prefix.split()]

        last_prefixes.append({
            'case_id': case_id,
            'token_ids': token_ids,
            'time_features': time_features,
            'prefix': prefix
        })
        batch_case_ids.append(case_id)

    print(f"Prepared {len(last_prefixes)} cases for inference")

    # Pad all sequences
    all_token_ids = [p['token_ids'] for p in last_prefixes]
    token_x = loader.pad_sequences(all_token_ids, maxlen=max_case_length)
    token_x = torch.tensor(token_x, dtype=torch.long)

    time_x = np.array([p['time_features'] for p in last_prefixes], dtype=np.float32)
    time_x = torch.tensor(time_x, dtype=torch.float32)

    # Create dataset and dataloader
    dataset = InferenceDataset(token_x, time_x, batch_case_ids)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Created DataLoader with batch_size={args.batch_size}")

    # Step 4: Run batch inference
    print("\n" + "=" * 70)
    print("Step 4: Running batch inference")
    print("=" * 70)

    all_predictions = []
    all_attention_scores = {}

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Processing batches"):
            batch_token_x = batch['token_x'].to(args.device)
            batch_time_x = batch['time_x'].to(args.device)
            batch_case_ids = batch['case_id']

            # Forward pass with attention
            predictions, attn_weights = model(batch_token_x, batch_time_x, return_attention=True)

            # Extract predictions
            batch_predictions = predictions[:, 0].cpu().numpy()

            # Inverse transform if scaler available
            if y_scaler is not None:
                batch_predictions = y_scaler.inverse_transform(
                    batch_predictions.reshape(-1, 1)
                ).flatten()

            # Aggregate attention scores for each sample in batch
            batch_attention = aggregate_attention_scores_batch(attn_weights, batch_token_x)

            # Store results
            for i, case_id in enumerate(batch_case_ids):
                all_predictions.append({
                    'case_id': case_id,
                    'prediction': float(batch_predictions[i])
                })
                all_attention_scores[case_id] = batch_attention[i]

    print(f"✓ Completed inference for {len(all_predictions)} cases")

    # Step 5: Convert attention scores to activity scores
    print("\n" + "=" * 70)
    print("Step 5: Computing activity attention scores")
    print("=" * 70)

    # Map token IDs back to activities
    id_to_activity = {int(v): k for k, v in x_word_dict.items()}

    # Create activity scores for all cases
    # All cases will have the same activity list (from y_word_dict)
    all_activities = list(y_word_dict.keys())
    case_activity_scores = {}

    for case_id in tqdm(case_ids, desc="Computing activity scores"):
        # Initialize all activities with 0
        activity_scores = {activity: 0.0 for activity in all_activities}

        # Get attention scores for this case (token_id -> score)
        token_scores = all_attention_scores.get(case_id, {})

        # Map token scores to activity scores
        for token_id, score in token_scores.items():
            if token_id in id_to_activity:
                activity = id_to_activity[token_id]
                # Only include activities that are in y_word_dict
                if activity in activity_scores:
                    activity_scores[activity] = float(score)

        case_activity_scores[case_id] = activity_scores

    print(f"✓ Computed activity scores for {len(case_activity_scores)} cases")

    # Step 6: Save results to JSON
    print("\n" + "=" * 70)
    print("Step 6: Saving results")
    print("=" * 70)

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    # Save predictions
    predictions_filename = f"{input_name}_predictions_{timestamp}.json"
    predictions_path = os.path.join(results_dir, predictions_filename)

    predictions_dict = {pred['case_id']: pred['prediction'] for pred in all_predictions}

    with open(predictions_path, 'w') as f:
        json.dump(predictions_dict, f, indent=2)

    print(f"✓ Predictions saved to: {predictions_path}")

    # Save attention scores
    attention_filename = f"{input_name}_attention_scores_{timestamp}.json"
    attention_path = os.path.join(results_dir, attention_filename)

    with open(attention_path, 'w') as f:
        json.dump(case_activity_scores, f, indent=2)

    print(f"✓ Attention scores saved to: {attention_path}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    pred_values = [p['prediction'] for p in all_predictions]
    print(f"Predictions:")
    print(f"  Min: {min(pred_values):.4f} days")
    print(f"  Max: {max(pred_values):.4f} days")
    print(f"  Mean: {np.mean(pred_values):.4f} days")
    print(f"  Median: {np.median(pred_values):.4f} days")

    print(f"\nAttention Scores:")
    print(f"  Cases processed: {len(case_activity_scores)}")
    print(f"  Activities per case: {len(all_activities)}")

    # Show sample for first case
    if case_ids:
        sample_case = case_ids[0]
        print(f"\nSample attention scores for case '{sample_case}':")
        sample_scores = case_activity_scores[sample_case]
        # Show only non-zero scores
        non_zero = {k: v for k, v in sample_scores.items() if v > 0}
        for activity, score in sorted(non_zero.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {activity}: {score:.6f}")

    print("\n" + "=" * 70)
    print("Batch inference completed successfully!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  Predictions: {predictions_path}")
    print(f"  Attention scores: {attention_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Transformer - Remaining Time Batch Inference\n"
                    "Process multiple cases from a single CSV file and generate attention scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--dataset", required=True, type=str,
                        help="dataset name")

    parser.add_argument("--input_csv", required=True, type=str,
                        help="input CSV filename (e.g., test_traces.csv). "
                             "Can contain multiple cases. "
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

    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size for inference (default: 32)")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device to run inference on")

    args = parser.parse_args()

    run_batch_inference(args)
