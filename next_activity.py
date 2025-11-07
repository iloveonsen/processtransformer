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

parser = argparse.ArgumentParser(description="Process Transformer - Next Activity Prediction.")

parser.add_argument("--dataset", required=True, type=str, help="dataset name")

parser.add_argument("--model_dir", default="./models", type=str, help="model directory")

parser.add_argument("--result_dir", default="./results", type=str, help="results directory")

parser.add_argument("--task", type=constants.Task,
    default=constants.Task.NEXT_ACTIVITY,  help="task name")

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
    model_path = f"{model_path}/next_activity_ckpt.pt"

    result_path = f"{args.result_dir}/{args.dataset}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = f"{result_path}/results"

    # Load data
    data_loader = loader.LogsDataLoader(name = args.dataset)

    (train_df, test_df, x_word_dict, y_word_dict, max_case_length,
        vocab_size, num_output) = data_loader.load_data(args.task)

    # Prepare training examples for next activity prediction task
    train_token_x, train_token_y = data_loader.prepare_data_next_activity(train_df,
        x_word_dict, y_word_dict, max_case_length)

    # Create and train a transformer model
    transformer_model = transformer.get_next_activity_model(
        max_case_length=max_case_length,
        vocab_size=vocab_size,
        output_dim=num_output)

    transformer_model = transformer_model.to(device)

    # Define optimizer and loss
    optimizer = optim.Adam(transformer_model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_accuracy = 0.0
    num_batches = len(train_token_x) // args.batch_size

    for epoch in range(args.epochs):
        transformer_model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        # Shuffle data
        indices = np.random.permutation(len(train_token_x))
        train_token_x = train_token_x[indices]
        train_token_y = train_token_y[indices]

        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = start_idx + args.batch_size

            batch_x = torch.tensor(train_token_x[start_idx:end_idx], dtype=torch.long).to(device)
            batch_y = torch.tensor(train_token_y[start_idx:end_idx], dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs = transformer_model(batch_x, training=True)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        accuracy = correct / total
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(transformer_model.state_dict(), model_path)
            print(f"Model saved with accuracy: {accuracy:.4f}")

    # Load best model for evaluation
    transformer_model.load_state_dict(torch.load(model_path))
    transformer_model.eval()

    # Evaluate over all the prefixes (k) and save the results
    k, accuracies, fscores, precisions, recalls = [], [], [], [], []

    with torch.no_grad():
        for i in range(max_case_length):
            test_data_subset = test_df[test_df["k"]==i]
            if len(test_data_subset) > 0:
                test_token_x, test_token_y = data_loader.prepare_data_next_activity(test_data_subset,
                    x_word_dict, y_word_dict, max_case_length)

                # Predict in batches
                all_predictions = []
                for batch_start in range(0, len(test_token_x), args.batch_size):
                    batch_end = min(batch_start + args.batch_size, len(test_token_x))
                    batch_x = torch.tensor(test_token_x[batch_start:batch_end], dtype=torch.long).to(device)
                    outputs = transformer_model(batch_x, training=False)
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_predictions.extend(predictions)

                y_pred = np.array(all_predictions)
                accuracy = metrics.accuracy_score(test_token_y, y_pred)
                precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
                    test_token_y, y_pred, average="weighted")
                k.append(i)
                accuracies.append(accuracy)
                fscores.append(fscore)
                precisions.append(precision)
                recalls.append(recall)

    k.append(i + 1)
    accuracies.append(np.mean(accuracies))
    fscores.append(np.mean(fscores))
    precisions.append(np.mean(precisions))
    recalls.append(np.mean(recalls))
    print('Average accuracy across all prefixes:', np.mean(accuracies))
    print('Average f-score across all prefixes:', np.mean(fscores))
    print('Average precision across all prefixes:', np.mean(precisions))
    print('Average recall across all prefixes:', np.mean(recalls))
    results_df = pd.DataFrame({"k":k, "accuracy":accuracies, "fscore": fscores,
        "precision":precisions, "recall":recalls})
    results_df.to_csv(result_path+"_next_activity.csv", index=False)
