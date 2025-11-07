import os
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics

from processtransformer import constants
from processtransformer.data import loader
from processtransformer.models import transformer

parser = argparse.ArgumentParser(description="Process Transformer - Remaining Time Prediction.")

parser.add_argument("--dataset", required=True, type=str, help="dataset name")

parser.add_argument("--model_dir", default="./models", type=str, help="model directory")

parser.add_argument("--result_dir", default="./results", type=str, help="results directory")

parser.add_argument("--task", type=constants.Task,
    default=constants.Task.REMAINING_TIME,  help="task name")

parser.add_argument("--epochs", default=10, type=int, help="number of total epochs")

parser.add_argument("--batch_size", default=12, type=int, help="batch size")

parser.add_argument("--learning_rate", default=0.001, type=float,
                    help="learning rate")

parser.add_argument("--gpu", default=0, type=int,
                    help="gpu id")

args = parser.parse_args()

if __name__ == "__main__":
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories to save the results and models
    model_path = f"{args.model_dir}/{args.dataset}"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = f"{model_path}/remaining_time_ckpt.pt"

    result_path = f"{args.result_dir}/{args.dataset}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = f"{result_path}/results"

    # Load data
    data_loader = loader.LogsDataLoader(name = args.dataset)

    (train_df, test_df, x_word_dict, y_word_dict, max_case_length,
        vocab_size, num_output) = data_loader.load_data(args.task)

    # Prepare training examples for next time prediction task
    (train_token_x, train_time_x,
        train_token_y, time_scaler, y_scaler) = data_loader.prepare_data_remaining_time(train_df,
        x_word_dict, max_case_length)

    # Create and train a transformer model
    transformer_model = transformer.get_remaining_time_model(
        max_case_length=max_case_length,
        vocab_size=vocab_size)

    transformer_model = transformer_model.to(device)

    # Define optimizer and loss
    optimizer = optim.Adam(transformer_model.parameters(), lr=args.learning_rate)
    # LogCosh loss implementation
    class LogCoshLoss(nn.Module):
        def __init__(self):
            super(LogCoshLoss, self).__init__()

        def forward(self, y_pred, y_true):
            loss = torch.log(torch.cosh(y_pred - y_true))
            return torch.mean(loss)

    criterion = LogCoshLoss()

    # Training loop
    best_loss = float('inf')
    num_batches = len(train_token_x) // args.batch_size

    for epoch in range(args.epochs):
        transformer_model.train()
        epoch_loss = 0.0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = start_idx + args.batch_size

            batch_x = torch.tensor(train_token_x[start_idx:end_idx], dtype=torch.long).to(device)
            batch_time_x = torch.tensor(train_time_x[start_idx:end_idx], dtype=torch.float32).to(device)
            batch_y = torch.tensor(train_token_y[start_idx:end_idx], dtype=torch.float32).to(device)

            optimizer.zero_grad()
            outputs = transformer_model(batch_x, batch_time_x, training=True)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(transformer_model.state_dict(), model_path)
            print(f"Model saved with loss: {avg_loss:.4f}")

    # Load best model for evaluation
    transformer_model.load_state_dict(torch.load(model_path))
    transformer_model.eval()

    # Evaluate over all the prefixes (k) and save the results
    k, maes, mses, rmses = [], [], [], []

    with torch.no_grad():
        for i in range(max_case_length):
            test_data_subset = test_df[test_df["k"]==i]
            if len(test_data_subset) > 0:
                test_token_x, test_time_x, test_y, _, _ = data_loader.prepare_data_remaining_time(
                    test_data_subset, x_word_dict, max_case_length, time_scaler, y_scaler, False)

                # Predict in batches
                all_predictions = []
                for batch_start in range(0, len(test_token_x), args.batch_size):
                    batch_end = min(batch_start + args.batch_size, len(test_token_x))
                    batch_x = torch.tensor(test_token_x[batch_start:batch_end], dtype=torch.long).to(device)
                    batch_time_x = torch.tensor(test_time_x[batch_start:batch_end], dtype=torch.float32).to(device)
                    outputs = transformer_model(batch_x, batch_time_x, training=False)
                    predictions = outputs.cpu().numpy()
                    all_predictions.append(predictions)

                y_pred = np.vstack(all_predictions)
                _test_y = y_scaler.inverse_transform(test_y)
                _y_pred = y_scaler.inverse_transform(y_pred)

                k.append(i)
                maes.append(metrics.mean_absolute_error(_test_y, _y_pred))
                mses.append(metrics.mean_squared_error(_test_y, _y_pred))
                rmses.append(np.sqrt(metrics.mean_squared_error(_test_y, _y_pred)))

    k.append(i + 1)
    maes.append(np.mean(maes))
    mses.append(np.mean(mses))
    rmses.append(np.mean(rmses))
    print('Average MAE across all prefixes:', np.mean(maes))
    print('Average MSE across all prefixes:', np.mean(mses))
    print('Average RMSE across all prefixes:', np.mean(rmses))
    results_df = pd.DataFrame({"k":k, "mean_absolute_error":maes,
        "mean_squared_error":mses,
        "root_mean_squared_error":rmses})
    results_df.to_csv(result_path+"_remaining_time.csv", index=False)
