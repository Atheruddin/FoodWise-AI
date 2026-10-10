import pandas as pd
import numpy as np

def create_time_series_features(full_df):
    """
    Computes time-series lags and rolling statistics per (center_id, meal_id).
    Strictly prevents target leakage by taking shifts before computing rolling windows.
    """
    df = full_df.copy()
    df = df.sort_values(by=["center_id", "meal_id", "week"]).reset_index(drop=True)
    
    group_cols = ["center_id", "meal_id"]
    
    # Lag features of num_orders
    if "num_orders" in df.columns:
        df["lag_1"] = df.groupby(group_cols)["num_orders"].shift(1)
        df["lag_2"] = df.groupby(group_cols)["num_orders"].shift(2)
        df["lag_3"] = df.groupby(group_cols)["num_orders"].shift(3)
        df["lag_4"] = df.groupby(group_cols)["num_orders"].shift(4)
        
        # Rolling statistics derived strictly from lag_1 (prior observations)
        df["rolling_mean_4"] = df.groupby(group_cols)["lag_1"].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
        df["rolling_std_4"] = df.groupby(group_cols)["lag_1"].transform(lambda x: x.rolling(window=4, min_periods=1).std()).fillna(0)
        df["rolling_mean_8"] = df.groupby(group_cols)["lag_1"].transform(lambda x: x.rolling(window=8, min_periods=1).mean())
        df["rolling_std_8"] = df.groupby(group_cols)["lag_1"].transform(lambda x: x.rolling(window=8, min_periods=1).std()).fillna(0)
        df["rolling_min_4"] = df.groupby(group_cols)["lag_1"].transform(lambda x: x.rolling(window=4, min_periods=1).min())
        df["rolling_max_4"] = df.groupby(group_cols)["lag_1"].transform(lambda x: x.rolling(window=4, min_periods=1).max())
    
    # Price lags and changes
    df["checkout_price_lag_1"] = df.groupby(group_cols)["checkout_price"].shift(1)
    df["price_diff_lag_1"] = df["checkout_price"] - df["checkout_price_lag_1"].fillna(df["checkout_price"])
    df["price_pct_change"] = (df["price_diff_lag_1"] / np.maximum(df["checkout_price_lag_1"].fillna(df["checkout_price"]), 1e-5)).fillna(0)
    
    return df

def apply_target_encodings(df, train_split_df, train_stats=None):
    """
    Applies historical demand statistics (target encoding) fitted strictly on the training split.
    Prevents data leakage into validation or test sets.
    """
    df = df.copy()
    
    if train_stats is None:
        train_stats = {}
        # Compute mean and std demand on training split only
        cat_stats = train_split_df.groupby("category")["num_orders"].agg(
            category_mean_orders="mean", category_std_orders="std"
        ).reset_index()
        
        cuisine_stats = train_split_df.groupby("cuisine")["num_orders"].agg(
            cuisine_mean_orders="mean", cuisine_std_orders="std"
        ).reset_index()
        
        center_stats = train_split_df.groupby("center_id")["num_orders"].agg(
            center_mean_orders="mean", center_std_orders="std"
        ).reset_index()
        
        meal_stats = train_split_df.groupby("meal_id")["num_orders"].agg(
            meal_mean_orders="mean", meal_std_orders="std"
        ).reset_index()
        
        center_meal_stats = train_split_df.groupby(["center_id", "meal_id"])["num_orders"].agg(
            center_meal_mean_orders="mean", center_meal_std_orders="std"
        ).reset_index()
        
        global_mean = float(train_split_df["num_orders"].mean())
        
        train_stats = {
            "cat_stats": cat_stats,
            "cuisine_stats": cuisine_stats,
            "center_stats": center_stats,
            "meal_stats": meal_stats,
            "center_meal_stats": center_meal_stats,
            "global_mean": global_mean
        }

    # Merge target encoding features onto df
    df = df.merge(train_stats["cat_stats"], on="category", how="left")
    df = df.merge(train_stats["cuisine_stats"], on="cuisine", how="left")
    df = df.merge(train_stats["center_stats"], on="center_id", how="left")
    df = df.merge(train_stats["meal_stats"], on="meal_id", how="left")
    df = df.merge(train_stats["center_meal_stats"], on=["center_id", "meal_id"], how="left")
    
    # Fill any unseen categories with global statistics
    df["category_mean_orders"] = df["category_mean_orders"].fillna(train_stats["global_mean"])
    df["category_std_orders"] = df["category_std_orders"].fillna(0)
    df["cuisine_mean_orders"] = df["cuisine_mean_orders"].fillna(train_stats["global_mean"])
    df["cuisine_std_orders"] = df["cuisine_std_orders"].fillna(0)
    df["center_mean_orders"] = df["center_mean_orders"].fillna(train_stats["global_mean"])
    df["center_std_orders"] = df["center_std_orders"].fillna(0)
    df["meal_mean_orders"] = df["meal_mean_orders"].fillna(train_stats["global_mean"])
    df["meal_std_orders"] = df["meal_std_orders"].fillna(0)
    df["center_meal_mean_orders"] = df["center_meal_mean_orders"].fillna(df["meal_mean_orders"])
    df["center_meal_std_orders"] = df["center_meal_std_orders"].fillna(0)

    return df, train_stats

def encode_categorical_features(df):
    """
    One-hot encodes categorical variables for model training.
    Converts booleans to int.
    """
    df = df.copy()
    categorical_cols = ["category", "cuisine", "center_type"]
    cols_to_encode = [c for c in categorical_cols if c in df.columns]
    
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=False)
    
    # Convert bool columns to int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    for c in bool_cols:
        df[c] = df[c].astype(int)
        
    return df

def get_feature_columns(df):
    """
    Returns the list of numerical feature columns used for model training.
    Excludes identifiers and target columns.
    """
    exclude = ["id", "num_orders", "log_num_orders", "category", "cuisine", "center_type", "city_code", "region_code"]
    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        # Check numeric or boolean
        if df[col].dtype.kind in "bif": # bool, int, float
            feature_cols.append(col)
    return feature_cols
