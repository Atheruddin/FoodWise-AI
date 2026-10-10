import pandas as pd
import numpy as np

def preprocess_dataframe(df):
    """
    Cleans and creates initial interaction features on merged dataframe.
    """
    df = df.copy()
    
    # Sort chronologically per center-meal pair
    df = df.sort_values(by=["center_id", "meal_id", "week"]).reset_index(drop=True)
    
    # Price difference and discount percentage
    df["discount_amount"] = df["base_price"] - df["checkout_price"]
    df["discount_percent"] = (df["discount_amount"] / np.maximum(df["base_price"], 1e-5)).clip(lower=0.0)
    df["price_ratio"] = (df["checkout_price"] / np.maximum(df["base_price"], 1e-5))
    
    # Promotion interaction feature
    df["promo_combo"] = df["emailer_for_promotion"] + 2 * df["homepage_featured"]
    
    # Ensure proper data types
    categorical_cols = ["category", "cuisine", "center_type", "city_code", "region_code"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            
    return df
