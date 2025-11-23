#!/usr/bin/env python3
"""
XES to CSV Converter for Process Transformer

This script converts XES event log files to CSV format compatible with
the data_processing.py script.

Usage:
    python xes_to_csv.py <input.xes>
    python xes_to_csv.py <input.xes> <output.csv>
    python xes_to_csv.py --dir <directory>  # Convert all XES files in directory

Requirements:
    - pm4py>=2.7.0
    - pandas

The output CSV will have the following columns required by data_processing.py:
    - case:concept:name (Case ID)
    - concept:name (Activity name)
    - time:timestamp (Timestamp in format: YYYY-MM-DD HH:MM:SS)
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

try:
    import pm4py
except ImportError:
    print("Error: pm4py is not installed.")
    print("Install it with: pip install pm4py")
    sys.exit(1)


def convert_xes_to_csv(xes_file_path, csv_file_path=None, verbose=True):
    """
    Convert a XES file to CSV format.

    Args:
        xes_file_path (str): Path to the input XES file
        csv_file_path (str, optional): Path to the output CSV file.
                                       If None, uses same name as XES with .csv extension
        verbose (bool): Print progress messages

    Returns:
        str: Path to the created CSV file

    Raises:
        FileNotFoundError: If XES file doesn't exist
        ValueError: If XES file cannot be parsed
    """
    xes_path = Path(xes_file_path)

    if not xes_path.exists():
        raise FileNotFoundError(f"XES file not found: {xes_file_path}")

    if verbose:
        print(f"Reading XES file: {xes_file_path}")

    try:
        # Read XES file - pm4py 2.7+ returns DataFrame by default
        df = pm4py.read_xes(str(xes_path))

        if verbose:
            print(f"✓ Loaded {len(df)} events from XES file")
            print(f"  Unique cases: {df['case:concept:name'].nunique()}")
            print(f"  Unique activities: {df['concept:name'].nunique()}")

    except Exception as e:
        raise ValueError(f"Failed to parse XES file: {e}")

    # Verify required columns exist
    required_columns = ['case:concept:name', 'concept:name', 'time:timestamp']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        if verbose:
            print(f"Warning: Missing columns {missing_columns}")
            print(f"Available columns: {list(df.columns)}")

        # Try to map common column names
        column_mapping = {}

        # Map case ID
        if 'case:concept:name' not in df.columns:
            for possible_case_col in ['case_id', 'caseid', 'case', 'Case ID']:
                if possible_case_col in df.columns:
                    column_mapping[possible_case_col] = 'case:concept:name'
                    break

        # Map activity name
        if 'concept:name' not in df.columns:
            for possible_activity_col in ['activity', 'Activity', 'event', 'Event']:
                if possible_activity_col in df.columns:
                    column_mapping[possible_activity_col] = 'concept:name'
                    break

        # Map timestamp
        if 'time:timestamp' not in df.columns:
            for possible_time_col in ['timestamp', 'Timestamp', 'time', 'Time', 'start_timestamp']:
                if possible_time_col in df.columns:
                    column_mapping[possible_time_col] = 'time:timestamp'
                    break

        if column_mapping:
            df = df.rename(columns=column_mapping)
            if verbose:
                print(f"✓ Mapped columns: {column_mapping}")

    # Select only required columns
    df = df[required_columns]

    # Ensure timestamp is in correct format
    if not pd.api.types.is_datetime64_any_dtype(df['time:timestamp']):
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

    # Format timestamp as string in required format
    df['time:timestamp'] = df['time:timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Determine output path
    if csv_file_path is None:
        csv_file_path = xes_path.with_suffix('.csv')

    csv_path = Path(csv_file_path)

    # Save to CSV
    df.to_csv(csv_path, index=False)

    if verbose:
        print(f"✓ Saved CSV file: {csv_path}")
        print(f"  File size: {csv_path.stat().st_size / 1024:.2f} KB")

    return str(csv_path)


def convert_directory(directory_path, output_dir=None, verbose=True):
    """
    Convert all XES files in a directory to CSV.

    Args:
        directory_path (str): Path to directory containing XES files
        output_dir (str, optional): Output directory for CSV files.
                                   If None, saves in same directory
        verbose (bool): Print progress messages

    Returns:
        list: List of paths to created CSV files
    """
    dir_path = Path(directory_path)

    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory_path}")

    xes_files = list(dir_path.glob("*.xes"))

    if not xes_files:
        print(f"No XES files found in {directory_path}")
        return []

    if verbose:
        print(f"Found {len(xes_files)} XES file(s) in {directory_path}")
        print()

    csv_files = []

    for xes_file in xes_files:
        try:
            if output_dir:
                output_path = Path(output_dir) / xes_file.with_suffix('.csv').name
                output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_path = None

            csv_path = convert_xes_to_csv(xes_file, output_path, verbose=verbose)
            csv_files.append(csv_path)

            if verbose:
                print()

        except Exception as e:
            print(f"✗ Failed to convert {xes_file.name}: {e}")
            if verbose:
                print()
            continue

    if verbose:
        print(f"{'='*60}")
        print(f"Conversion complete: {len(csv_files)}/{len(xes_files)} files converted successfully")

    return csv_files


def main():
    parser = argparse.ArgumentParser(
        description="Convert XES event log files to CSV format for Process Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file (output: same name with .csv extension)
  python xes_to_csv.py input.xes

  # Convert single file with custom output name
  python xes_to_csv.py input.xes output.csv

  # Convert all XES files in a directory
  python xes_to_csv.py --dir ./xes_logs/

  # Convert all XES files to a different directory
  python xes_to_csv.py --dir ./xes_logs/ --output ./csv_logs/
        """
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="Input XES file path"
    )

    parser.add_argument(
        "output",
        nargs="?",
        help="Output CSV file path (optional)"
    )

    parser.add_argument(
        "--dir",
        dest="directory",
        help="Convert all XES files in directory"
    )

    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Output directory for CSV files (used with --dir)"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    verbose = not args.quiet

    try:
        # Directory mode
        if args.directory:
            csv_files = convert_directory(
                args.directory,
                output_dir=args.output_dir,
                verbose=verbose
            )

            if not csv_files:
                sys.exit(1)

        # Single file mode
        elif args.input:
            convert_xes_to_csv(args.input, args.output, verbose=verbose)

        else:
            parser.print_help()
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
