import os
import json
import argparse
import time
import numpy as np
import pandas as pd

from processtransformer import constants
from processtransformer.data.processor import LogsDataProcessor

# Load datasets configuration
datasets_config_path = os.path.join(os.path.dirname(__file__), "config", "datasets.json")
with open(datasets_config_path, "r") as f:
    datasets_config = json.load(f)

parser = argparse.ArgumentParser(
    description="Process Transformer - Data Processing.")

parser.add_argument("--dataset",
    type=str,
    default="helpdesk",
    help="dataset name (must be defined in config/datasets.json)")

parser.add_argument("--dir_path",
    type=str,
    default="./datasets",
    help="base data directory path")

parser.add_argument("--raw_log_file",
    type=str,
    default=None,
    help="path to raw csv log file (optional, overrides config/datasets.json)")

parser.add_argument("--columns",
    type=str,
    nargs='+',
    default=None,
    help="column names (optional, overrides config/datasets.json)")

parser.add_argument("--task",
    type=constants.Task,
    default=constants.Task.REMAINING_TIME,
    help="task name")

parser.add_argument("--sort_temporally",
    type=bool,
    default=False,
    help="sort cases by timestamp")

args = parser.parse_args()

if __name__ == "__main__":
    """
    Expected directory structure:
    ./datasets/
    ├── raw/
    │   └── {dataset_name}.csv  (e.g., helpdesk.csv)
    └── processed/
        └── {dataset_name}/  (created automatically)
            ├── metadata.json
            ├── next_activity_train.csv
            ├── next_activity_test.csv
            ├── next_time_train.csv
            ├── next_time_test.csv
            ├── remaining_time_train.csv
            └── remaining_time_test.csv
    """

    # Validate dataset exists in config
    if args.dataset not in datasets_config:
        raise ValueError(
            f"Dataset '{args.dataset}' not found in config/datasets.json. "
            f"Available datasets: {list(datasets_config.keys())}"
        )

    # Get dataset configuration
    dataset_info = datasets_config[args.dataset]

    # Use config values, but allow command-line overrides
    raw_log_file = args.raw_log_file if args.raw_log_file else f"{args.dir_path}/raw/{dataset_info['raw_file']}"
    columns = args.columns if args.columns else dataset_info['columns']

    print(f"Processing dataset: {args.dataset}")
    print(f"  Description: {dataset_info['description']}")
    print(f"  Raw file: {raw_log_file}")
    print(f"  Columns: {columns}")
    print(f"  Task: {args.task.value}")
    print()

    # Process raw logs
    start = time.time()
    data_processor = LogsDataProcessor(
        name=args.dataset,
        filepath=raw_log_file,
        columns=columns,
        dir_path=args.dir_path,
        pool=1
    )
    data_processor.process_logs(task=args.task, sort_temporally=args.sort_temporally)
    end = time.time()
    print(f"\nTotal processing time: {end - start:.2f} seconds")

