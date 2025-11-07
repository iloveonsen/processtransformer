import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn import metrics

from processtransformer import constants
from processtransformer.data import loader
from processtransformer.models import transformer

# Load default configuration
config_path = os.path.join(os.path.dirname(__file__), "config", "next_time.json")
with open(config_path, "r") as f:
    config = json.load(f)

parser = argparse.ArgumentParser(description="Process Transformer - Next Time Prediction.")

parser.add_argument("--dataset", required=True, type=str, help="dataset name")

parser.add_argument("--model_dir", default=config["model_dir"], type=str, help="model directory")

parser.add_argument("--result_dir", default=config["result_dir"], type=str, help="results directory")

parser.add_argument("--task", type=constants.Task,
    default=constants.Task.NEXT_TIME,  help="task name")

parser.add_argument("--epochs", default=config["epochs"], type=int, help="number of total epochs")

parser.add_argument("--batch_size", default=config["batch_size"], type=int, help="batch size")

parser.add_argument("--learning_rate", default=config["learning_rate"], type=float,
                    help="learning rate")

parser.add_argument("--gpu", default=config["gpu"], type=str,
                    help="gpu ids (comma-separated, e.g., '0,1' for 2 GPUs)")

parser.add_argument("--num_workers", default=config["num_workers"], type=int,
                    help="number of data loading workers")

args = parser.parse_args()

# LogCosh loss implementation
class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = torch.log(torch.cosh(y_pred - y_true))
        return torch.mean(loss)

if __name__ == "__main__":
    # Set device(s)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpu_ids = [int(i) for i in args.gpu.split(',') if i.strip()]
    device = torch.device("cuda:0" if torch.cuda.is_available() and len(gpu_ids) > 0 else "cpu")
    device_type = "cuda" if torch.cuda.is_available() and len(gpu_ids) > 0 else "cpu"
    use_multi_gpu = torch.cuda.is_available() and len(gpu_ids) > 1

    print(f"Using device: {device}")
    if use_multi_gpu:
        print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")

    # Create directories to save the results and models
    model_path = f"{args.model_dir}/{args.dataset}"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    checkpoint_path = f"{model_path}/next_time_ckpt.pt"

    result_path = f"{args.result_dir}/{args.dataset}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = f"{result_path}/results"

    # Load data
    data_loader = loader.LogsDataLoader(name=args.dataset)

    (train_df, test_df, x_word_dict, y_word_dict, max_case_length,
        vocab_size, num_output) = data_loader.load_data(args.task)

    # Prepare training examples for next time prediction task
    (train_token_x, train_time_x,
        train_token_y, time_scaler, y_scaler) = data_loader.prepare_data_next_time(train_df,
        x_word_dict, max_case_length)

    # Create PyTorch Dataset and DataLoader
    train_dataset = loader.TimeDataset(train_token_x, train_time_x, train_token_y)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Create and train a transformer model
    transformer_model = transformer.get_next_time_model(
        max_case_length=max_case_length,
        vocab_size=vocab_size)

    transformer_model = transformer_model.to(device)

    # Multi-GPU support
    if use_multi_gpu:
        transformer_model = nn.DataParallel(transformer_model)
        print(f"Model wrapped with DataParallel")

    # Define optimizer, loss, and scheduler
    optimizer = optim.Adam(transformer_model.parameters(), lr=args.learning_rate)
    criterion = LogCoshLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Mixed Precision Training
    scaler = GradScaler(device_type) if device_type == "cuda" else None

    # Training loop
    best_mae = float('inf')

    for epoch in range(args.epochs):
        transformer_model.train()
        epoch_loss = 0.0

        for batch_idx, (batch_x, batch_time_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_time_x = batch_time_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            if scaler is not None:
                # Mixed precision training
                with autocast(device_type=device_type):
                    outputs = transformer_model(batch_x, batch_time_x)
                    loss = criterion(outputs, batch_y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                outputs = transformer_model(batch_x, batch_time_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # Validation on training data to compute MAE and RMSE
        transformer_model.eval()
        all_predictions = []
        all_targets = []
        with torch.inference_mode():
            for batch_x, batch_time_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_time_x = batch_time_x.to(device)
                outputs = transformer_model(batch_x, batch_time_x)
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(batch_y.numpy())

        y_pred_scaled = np.vstack(all_predictions)
        y_true_scaled = np.vstack(all_targets)

        # Inverse transform to get actual values
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        y_true = y_scaler.inverse_transform(y_true_scaled)

        train_mae = metrics.mean_absolute_error(y_true, y_pred)
        train_rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f} - MAE: {train_mae:.4f} - RMSE: {train_rmse:.4f} - LR: {current_lr:.6f}")

        # Update learning rate scheduler
        scheduler.step(train_mae)

        # Save best model with checkpoint
        if train_mae < best_mae:
            best_mae = train_mae
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': transformer_model.module.state_dict() if use_multi_gpu else transformer_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'mae': train_mae,
                'rmse': train_rmse,
                'loss': avg_loss,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved with MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}")

        transformer_model.train()

    # Load best model for evaluation
    checkpoint = torch.load(checkpoint_path)
    if use_multi_gpu:
        transformer_model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        transformer_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best model loaded from epoch {checkpoint['epoch']+1} with MAE: {checkpoint['mae']:.4f}, RMSE: {checkpoint['rmse']:.4f}")

    transformer_model.eval()

    # Evaluate over all the prefixes (k) and save the results
    k, maes, mses, rmses = [], [], [], []

    with torch.inference_mode():
        for i in range(max_case_length):
            test_data_subset = test_df[test_df["k"]==i]
            if len(test_data_subset) > 0:
                test_token_x, test_time_x, test_y, _, _ = data_loader.prepare_data_next_time(
                    test_data_subset, x_word_dict, max_case_length, time_scaler, y_scaler, False)

                test_dataset = loader.TimeDataset(test_token_x, test_time_x, test_y)
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True if torch.cuda.is_available() else False
                )

                # Predict
                all_predictions = []
                all_targets = []
                for batch_x, batch_time_x, batch_y in test_loader:
                    batch_x = batch_x.to(device)
                    batch_time_x = batch_time_x.to(device)
                    outputs = transformer_model(batch_x, batch_time_x)
                    all_predictions.append(outputs.cpu().numpy())
                    all_targets.append(batch_y.numpy())

                y_pred = np.vstack(all_predictions)
                y_true = np.vstack(all_targets)

                _test_y = y_scaler.inverse_transform(y_true)
                _y_pred = y_scaler.inverse_transform(y_pred)

                k.append(i)
                maes.append(metrics.mean_absolute_error(_test_y, _y_pred))
                mses.append(metrics.mean_squared_error(_test_y, _y_pred))
                rmses.append(np.sqrt(metrics.mean_squared_error(_test_y, _y_pred)))

    k.append(len(maes))
    maes.append(np.mean(maes))
    mses.append(np.mean(mses))
    rmses.append(np.mean(rmses))
    print('Average MAE across all prefixes:', np.mean(maes[:-1]))
    print('Average MSE across all prefixes:', np.mean(mses[:-1]))
    print('Average RMSE across all prefixes:', np.mean(rmses[:-1]))
    results_df = pd.DataFrame({"k":k, "mean_absolute_error":maes,
        "mean_squared_error":mses,
        "root_mean_squared_error":rmses})
    results_df.to_csv(result_path+"_next_time.csv", index=False)
