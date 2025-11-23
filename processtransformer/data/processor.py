import os
import json
import pandas as pd
import numpy as np
import datetime
from tqdm.auto import tqdm

from ..constants import Task

class LogsDataProcessor:
    def __init__(self, name, filepath, columns, dir_path = "./datasets", pool = 1):
        """Provides support for processing raw logs.
        Args:
            name: str: Dataset name
            filepath: str: Path to raw logs dataset
            columns: list: name of column names
            dir_path:  str: Base directory path (defaults to ./datasets)
            pool: Number of CPUs (processes) to be used for data processing (deprecated, kept for compatibility)
        """
        self._name = name
        self._filepath = filepath
        self._org_columns = columns
        self._base_dir = dir_path
        # Create processed directory structure: ./datasets/processed/{name}/
        self._dir_path = f"{dir_path}/processed/{self._name}"
        if not os.path.exists(self._dir_path):
            os.makedirs(self._dir_path)
        self._pool = pool

    def _load_df(self, sort_temporally = False):
        df = pd.read_csv(self._filepath)
        df = df[self._org_columns]
        df.columns = ["case:concept:name", 
            "concept:name", "time:timestamp"]
        df["concept:name"] = df["concept:name"].str.lower()
        df["concept:name"] = df["concept:name"].str.replace(" ", "-")
        df["time:timestamp"] = df["time:timestamp"].str.replace("/", "-")
        df["time:timestamp"]= pd.to_datetime(df["time:timestamp"],
            format="%Y-%m-%d %H:%M:%S").map(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        if sort_temporally:
            df.sort_values(by = ["time:timestamp"], inplace = True)
        return df

    def _extract_logs_metadata(self, df):
        keys = ["[PAD]", "[UNK]"]
        activities = list(df["concept:name"].unique())
        keys.extend(activities)
        val = range(len(keys))

        coded_activity = dict({"x_word_dict":dict(zip(keys, val))})
        code_activity_normal = dict({"y_word_dict": dict(zip(activities, range(len(activities))))})

        coded_activity.update(code_activity_normal)
        coded_json = json.dumps(coded_activity)
        with open(f"{self._dir_path}/metadata.json", "w") as metadata_file:
            metadata_file.write(coded_json)

    def _next_activity_helper_func(self, df):
        case_id, case_name = "case:concept:name", "concept:name"
        processed_data = []

        # Use groupby to avoid repeated filtering - much faster!
        for case, group in tqdm(df.groupby(case_id), desc="Processing next activity", leave=False):
            act = group[case_name].to_list()
            for i in range(len(act) - 1):
                prefix = act[0] if i == 0 else " ".join(act[:i+1])
                next_act = act[i+1]
                processed_data.append({
                    "case_id": case,
                    "prefix": prefix,
                    "k": i,
                    "next_act": next_act
                })

        return pd.DataFrame(processed_data)

    def _process_next_activity(self, df, train_list, test_list):
        print("Processing next activity data...")
        processed_df = self._next_activity_helper_func(df)
        train_df = processed_df[processed_df["case_id"].isin(train_list)]
        test_df = processed_df[processed_df["case_id"].isin(test_list)]
        train_df.to_csv(f"{self._dir_path}/{Task.NEXT_ACTIVITY.value}_train.csv", index = False)
        test_df.to_csv(f"{self._dir_path}/{Task.NEXT_ACTIVITY.value}_test.csv", index = False)
        print(f"Saved {len(train_df)} training and {len(test_df)} test samples.")

    def _next_time_helper_func(self, df):
        case_id = "case:concept:name"
        event_name = "concept:name"
        event_time = "time:timestamp"
        processed_data = []

        # Use groupby to avoid repeated filtering
        for case, group in tqdm(df.groupby(case_id), desc="Processing next time", leave=False):
            act = group[event_name].to_list()
            time_str = group[event_time].str[:19].to_list()

            # Pre-parse all timestamps once - huge performance gain!
            time_dt = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in time_str]

            time_passed = 0

            for i in range(0, len(act)):
                prefix = act[0] if i == 0 else " ".join(act[:i+1])

                if i > 0:
                    latest_diff = time_dt[i] - time_dt[i-1]
                    latest_time = latest_diff.days
                else:
                    latest_time = 0

                if i > 1:
                    recent_diff = time_dt[i] - time_dt[i-2]
                    recent_time = recent_diff.days
                else:
                    recent_time = 0

                time_passed = time_passed + latest_time

                if i+1 < len(time_dt):
                    next_time = time_dt[i+1] - time_dt[i]
                    next_time_days = str(int(next_time.days))
                else:
                    next_time_days = str(1)

                processed_data.append({
                    "case_id": case,
                    "prefix": prefix,
                    "k": i,
                    "time_passed": time_passed,
                    "recent_time": recent_time,
                    "latest_time": latest_time,
                    "next_time": next_time_days
                })

        return pd.DataFrame(processed_data)

    def _process_next_time(self, df, train_list, test_list):
        print("Processing next time data...")
        processed_df = self._next_time_helper_func(df)
        train_df = processed_df[processed_df["case_id"].isin(train_list)]
        test_df = processed_df[processed_df["case_id"].isin(test_list)]
        train_df.to_csv(f"{self._dir_path}/{Task.NEXT_TIME.value}_train.csv", index = False)
        test_df.to_csv(f"{self._dir_path}/{Task.NEXT_TIME.value}_test.csv", index = False)
        print(f"Saved {len(train_df)} training and {len(test_df)} test samples.")

    def _remaining_time_helper_func(self, df):
        case_id = "case:concept:name"
        event_name = "concept:name"
        event_time = "time:timestamp"
        processed_data = []

        # Use groupby to avoid repeated filtering
        for case, group in tqdm(df.groupby(case_id), desc="Processing remaining time", leave=False):
            act = group[event_name].to_list()
            time_str = group[event_time].str[:19].to_list()

            # Pre-parse all timestamps once - huge performance gain!
            time_dt = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in time_str]

            time_passed = 0
            last_time_dt = time_dt[-1]  # Cache the last timestamp

            for i in range(0, len(act)):
                prefix = act[0] if i == 0 else " ".join(act[:i+1])

                if i > 0:
                    latest_diff = time_dt[i] - time_dt[i-1]
                    latest_time = latest_diff.days
                else:
                    latest_time = 0

                if i > 1:
                    recent_diff = time_dt[i] - time_dt[i-2]
                    recent_time = recent_diff.days
                else:
                    recent_time = 0

                time_passed = time_passed + latest_time

                # Calculate remaining time to case completion
                ttc = last_time_dt - time_dt[i]
                ttc = str(ttc.days)

                processed_data.append({
                    "case_id": case,
                    "prefix": prefix,
                    "k": i,
                    "time_passed": time_passed,
                    "recent_time": recent_time,
                    "latest_time": latest_time,
                    "remaining_time_days": ttc
                })

        return pd.DataFrame(processed_data)

    def _process_remaining_time(self, df, train_list, test_list):
        print("Processing remaining time data...")
        processed_df = self._remaining_time_helper_func(df)
        train_remaining_time = processed_df[processed_df["case_id"].isin(train_list)]
        test_remaining_time = processed_df[processed_df["case_id"].isin(test_list)]
        train_remaining_time.to_csv(f"{self._dir_path}/{Task.REMAINING_TIME.value}_train.csv", index = False)
        test_remaining_time.to_csv(f"{self._dir_path}/{Task.REMAINING_TIME.value}_test.csv", index = False)
        print(f"Saved {len(train_remaining_time)} training and {len(test_remaining_time)} test samples.")

    def process_logs(self, task, 
        sort_temporally = False, 
        train_test_ratio = 0.80):
        df = self._load_df(sort_temporally)
        self._extract_logs_metadata(df)
        train_test_ratio = int(abs(df["case:concept:name"].nunique()*train_test_ratio))
        train_list = df["case:concept:name"].unique()[:train_test_ratio]
        test_list = df["case:concept:name"].unique()[train_test_ratio:]
        if task == Task.NEXT_ACTIVITY:
            self._process_next_activity(df, train_list, test_list)
        elif task == Task.NEXT_TIME:
            self._process_next_time(df, train_list, test_list)
        elif task == Task.REMAINING_TIME:
            self._process_remaining_time(df, train_list, test_list)
        else:
            raise ValueError("Invalid task.")