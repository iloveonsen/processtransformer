# XES to CSV Converter

This directory contains utilities for converting XES (eXtensible Event Stream) files to CSV format compatible with the Process Transformer data processing pipeline.

## Quick Start

### Convert a single XES file

```bash
cd converters
python xes_to_csv.py ../datasets/raw/your_log.xes
```

This creates `your_log.csv` in the same directory.

### Convert all XES files in a directory

```bash
python xes_to_csv.py --dir ../datasets/raw/
```

This converts all `.xes` files to `.csv` in the same directory.

### Specify output location

```bash
python xes_to_csv.py input.xes ../datasets/raw/output.csv
```

or for batch conversion:

```bash
python xes_to_csv.py --dir ./xes_files/ --output-dir ../datasets/raw/
```

## Requirements

The converter requires `pm4py` library:

```bash
pip install pm4py
```

Or install all dependencies:

```bash
cd ..
pip install -e .
```

## Output Format

The converter produces CSV files with the following columns required by `data_processing.py`:

| Column | Description | Example |
|--------|-------------|---------|
| `case:concept:name` | Unique case/process instance ID | "Case_001" |
| `concept:name` | Activity/event name | "Submit Application" |
| `time:timestamp` | Timestamp in YYYY-MM-DD HH:MM:SS format | "2023-01-15 14:30:00" |

## Usage Examples

### Basic Usage

```bash
# Convert single file
python xes_to_csv.py my_process_log.xes

# Output: my_process_log.csv (in same directory)
```

### With Custom Output Path

```bash
python xes_to_csv.py source.xes ../datasets/raw/helpdesk.csv
```

### Batch Conversion

```bash
# Convert all XES files in a folder
python xes_to_csv.py --dir /path/to/xes/files/

# Convert to different output directory
python xes_to_csv.py --dir ./xes_logs/ --output-dir ../datasets/raw/
```

### Quiet Mode

```bash
# Suppress progress messages
python xes_to_csv.py --quiet input.xes
```

## Integration with Data Processing

After converting XES to CSV, you can process the data using the main pipeline:

```bash
cd ..
python data_processing.py --dataset my_dataset \
    --raw_log_file datasets/raw/my_process_log.csv \
    --columns "case:concept:name" "concept:name" "time:timestamp" \
    --task remaining_time
```

## Column Mapping

If your XES file uses non-standard column names, the converter will attempt to automatically map them:

**Case ID:** `case_id`, `caseid`, `case`, `Case ID` → `case:concept:name`
**Activity:** `activity`, `Activity`, `event`, `Event` → `concept:name`
**Timestamp:** `timestamp`, `Timestamp`, `time`, `Time`, `start_timestamp` → `time:timestamp`

If automatic mapping fails, you may need to manually adjust the XES file or modify the converter.

## Troubleshooting

### "pm4py is not installed"

Install pm4py:
```bash
pip install pm4py
```

### "Missing columns" warning

The XES file may use different attribute names. Check the warning message for available columns and either:
1. Let the automatic mapper try to fix it
2. Manually map columns in the converter script
3. Edit your XES file to use standard attribute names

### Timestamp format issues

The converter automatically converts timestamps to `YYYY-MM-DD HH:MM:SS` format. If you encounter issues, ensure your XES file has valid timestamp attributes.

## Supported XES Formats

The converter supports:
- Standard XES format (XES 1.0)
- XES 2.0 format
- Compressed XES files (.xes.gz)

## Advanced Usage

### Programmatic Usage

You can also use the converter as a Python module:

```python
from converters.xes_to_csv import convert_xes_to_csv, convert_directory

# Convert single file
csv_path = convert_xes_to_csv("input.xes", "output.csv")
print(f"Created: {csv_path}")

# Convert directory
csv_files = convert_directory("./xes_logs/", output_dir="./csv_logs/")
print(f"Converted {len(csv_files)} files")
```

## Common Workflow

1. **Place XES files** in `datasets/raw/`
   ```bash
   cp your_logs.xes datasets/raw/
   ```

2. **Convert to CSV**
   ```bash
   cd converters
   python xes_to_csv.py ../datasets/raw/your_logs.xes
   ```

3. **Add dataset config** to `config/datasets.json`
   ```json
   "your_logs": {
       "description": "Your process logs",
       "columns": ["case:concept:name", "concept:name", "time:timestamp"],
       "raw_file": "your_logs.csv"
   }
   ```

4. **Process the data**
   ```bash
   cd ..
   python data_processing.py --dataset your_logs --task remaining_time
   ```

5. **Train model**
   ```bash
   python remaining_time.py --dataset your_logs
   ```

## Notes

- The converter preserves all events and their order from the XES file
- Case IDs and activity names are used as-is from the XES file
- Timestamps are converted to string format for CSV compatibility
- The output CSV is compatible with both `data_processing.py` and pandas

## Support

For issues specific to:
- **XES parsing**: Check [pm4py documentation](https://pm4py.fit.fraunhofer.de/)
- **Data processing**: See main repository README
- **Converter bugs**: File an issue with sample XES file
