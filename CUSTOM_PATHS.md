# Custom Data Directory Configuration

This guide explains how to use custom data directories instead of the default `./datasets` path.

## Directory Structure

For any custom base path, the expected structure is:

```
{custom_dir}/
├── raw/
│   └── {dataset_name}.csv
└── processed/
    └── {dataset_name}/
        ├── metadata.json
        ├── next_activity_train.csv
        ├── next_activity_test.csv
        ├── next_time_train.csv
        ├── next_time_test.csv
        ├── remaining_time_train.csv
        └── remaining_time_test.csv
```

## Configuration

### Option 1: Using Config Files (Recommended)

Edit the configuration files in `config/` directory:

```json
{
  "data_dir": "/your/custom/path",
  "model_dir": "./models",
  "result_dir": "./results",
  ...
}
```

Files to edit:
- `config/next_activity.json`
- `config/next_time.json`
- `config/remaining_time.json`

### Option 2: Using Command Line Arguments

Override the default path using `--data_dir` argument.

## Usage Examples

### 1. Data Processing

```bash
# Using default ./datasets
python data_processing.py \
    --dataset helpdesk \
    --raw_log_file ./datasets/raw/helpdesk.csv \
    --task NEXT_ACTIVITY

# Using custom path
python data_processing.py \
    --dataset helpdesk \
    --raw_log_file /mnt/storage/ml_data/raw/helpdesk.csv \
    --dir_path /mnt/storage/ml_data \
    --task NEXT_ACTIVITY
```

### 2. Training - Next Activity Prediction

```bash
# Using default ./datasets (from config)
python next_activity.py --dataset helpdesk

# Using custom path via command line
python next_activity.py \
    --dataset helpdesk \
    --data_dir /mnt/storage/ml_data
```

### 3. Training - Next Time Prediction

```bash
# Using default ./datasets (from config)
python next_time.py --dataset bpi_2012

# Using custom path via command line
python next_time.py \
    --dataset bpi_2012 \
    --data_dir /home/user/process_mining_data
```

### 4. Training - Remaining Time Prediction

```bash
# Using default ./datasets (from config)
python remaining_time.py --dataset bpi_2013

# Using custom path via command line
python remaining_time.py \
    --dataset bpi_2013 \
    --data_dir /data/ml/process_logs
```

## Complete Workflow Example

```bash
# 1. Set custom data directory
CUSTOM_DIR="/mnt/storage/process_mining"

# 2. Create directory structure
mkdir -p ${CUSTOM_DIR}/raw
mkdir -p ${CUSTOM_DIR}/processed

# 3. Place raw data file
cp helpdesk.csv ${CUSTOM_DIR}/raw/

# 4. Process the data
python data_processing.py \
    --dataset helpdesk \
    --raw_log_file ${CUSTOM_DIR}/raw/helpdesk.csv \
    --dir_path ${CUSTOM_DIR} \
    --task NEXT_ACTIVITY

# 5. Train the model
python next_activity.py \
    --dataset helpdesk \
    --data_dir ${CUSTOM_DIR} \
    --epochs 20 \
    --batch_size 16

# Output:
# - Processed data: ${CUSTOM_DIR}/processed/helpdesk/
# - Model checkpoint: ./models/helpdesk/next_activity_ckpt.pt
# - Results: ./results/helpdesk/results_next_activity.csv
```

## Path Consistency Verification

The codebase ensures path consistency between data processing and loading:

- **LogsDataProcessor** writes to: `{dir_path}/processed/{dataset_name}/`
- **LogsDataLoader** reads from: `{dir_path}/processed/{dataset_name}/`

You can verify this by running:

```bash
python verify_custom_paths.py
```

## Advantages of Custom Paths

1. **Shared Storage**: Use network-attached storage for multiple machines
2. **Disk Space**: Use larger storage volumes for big datasets
3. **Organization**: Keep data separate from code repository
4. **Collaboration**: Share common data directory across team members

## Troubleshooting

### Path not found error

Make sure:
1. The base directory exists
2. The raw data file is in `{custom_dir}/raw/`
3. The dataset name matches the filename (without .csv extension)

### Permission errors

Ensure you have read/write permissions:
```bash
chmod -R u+rw /your/custom/path
```

### Relative vs Absolute Paths

- Relative paths: `./datasets`, `../shared_data`
- Absolute paths: `/home/user/data`, `/mnt/storage/ml_data`

Both work correctly. Absolute paths are recommended for production environments.
