import os
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")

def get_raw_file_paths():
    return {
        "train": os.path.join(RAW_DIR, "train.csv"),
        "test": os.path.join(RAW_DIR, "test.csv"),
        "meal_info": os.path.join(RAW_DIR, "meal_info.csv"),
        "center_info": os.path.join(RAW_DIR, "fulfilment_center_info.csv"),
        "sample_sub": os.path.join(RAW_DIR, "sample_submission.csv"),
    }

def load_raw_data():
    paths = get_raw_file_paths()
    data = {}
    for key, path in paths.items():
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
        else:
            data[key] = None
    return data

def load_merged_data():
    raw = load_raw_data()
    train_df = raw.get("train")
    test_df = raw.get("test")
    meal_df = raw.get("meal_info")
    center_df = raw.get("center_info")

    if train_df is None or meal_df is None or center_df is None:
        # Check if processed data exists
        proc_train_path = os.path.join(PROCESSED_DIR, "merged_train.parquet")
        if os.path.exists(proc_train_path):
            merged_train = pd.read_parquet(proc_train_path)
            proc_test_path = os.path.join(PROCESSED_DIR, "merged_test.parquet")
            merged_test = pd.read_parquet(proc_test_path) if os.path.exists(proc_test_path) else None
            return merged_train, merged_test, meal_df, center_df
        else:
            raise FileNotFoundError("Raw Kaggle datasets not found in data/raw/ and no processed files found.")

    # Merge train metadata
    merged_train = train_df.merge(meal_df, on="meal_id", how="left")
    merged_train = merged_train.merge(center_df, on="center_id", how="left")

    # Merge test metadata if test_df is available
    merged_test = None
    if test_df is not None:
        merged_test = test_df.merge(meal_df, on="meal_id", how="left")
        merged_test = merged_test.merge(center_df, on="center_id", how="left")

    return merged_train, merged_test, meal_df, center_df

def inspect_dataset_inventory():
    paths = get_raw_file_paths()
    inventory = []
    for key, path in paths.items():
        exists = os.path.exists(path)
        if exists:
            size_mb = round(os.path.getsize(path) / (1024 * 1024), 2)
            df = pd.read_csv(path)
            inventory.append({
                "File Key": key,
                "File Name": os.path.basename(path),
                "Exists": True,
                "Size (MB)": size_mb,
                "Rows": df.shape[0],
                "Columns": df.shape[1],
                "Column Names": ", ".join(df.columns),
                "Missing Values": df.isnull().sum().sum(),
                "Duplicates": df.duplicated().sum()
            })
        else:
            inventory.append({
                "File Key": key,
                "File Name": os.path.basename(path),
                "Exists": False,
                "Size (MB)": 0.0,
                "Rows": 0,
                "Columns": 0,
                "Column Names": "N/A",
                "Missing Values": 0,
                "Duplicates": 0
            })
    return pd.DataFrame(inventory)
