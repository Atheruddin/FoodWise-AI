import os
import json
import joblib
import pandas as pd
import numpy as np

from src.data_loader import PROCESSED_DIR
from src.preprocessing import preprocess_dataframe
from src.feature_engineering import encode_categorical_features

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def load_forecasting_artifacts():
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None, None
        
    model_data = joblib.load(model_path)
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
        
    return model_data, meta_data

def get_compact_dataset():
    compact_path = os.path.join(PROCESSED_DIR, "compact_features.parquet")
    if os.path.exists(compact_path):
        return pd.read_parquet(compact_path)
    return None

def generate_multi_week_forecast(center_id, meal_id, horizon_weeks=4):
    """
    Performs recursive multi-week demand forecasting for a specific center-meal pair.
    Uses model predictions recursively to update lag and rolling features for future weeks.
    
    Calculates Empirical Prediction Intervals (95%) using the holdout validation RMSE
    scaled recursively by horizon step sqrt(h).
    """
    artifacts, metadata = load_forecasting_artifacts()
    if artifacts is None:
        return None, "Model artifacts not found. Please ensure best_model.joblib exists under models/."
        
    model = artifacts["model"]
    feature_cols = artifacts["feature_cols"]
    train_stats = artifacts["full_train_stats"]
    val_rmse = float(artifacts.get("val_rmse", metadata.get("best_val_rmse", 114.70)))
    
    df_compact = get_compact_dataset()
    if df_compact is None:
        return None, "Compact processed dataset not found in data/processed/."
        
    # Filter historical data for center_id and meal_id
    sub = df_compact[(df_compact["center_id"] == center_id) & (df_compact["meal_id"] == meal_id)].copy()
    if len(sub) == 0:
        return None, f"No historical records found for Center {center_id} and Meal {meal_id}."
        
    sub = sub.sort_values(by="week").reset_index(drop=True)
    
    # Identify last available training week
    train_history = sub[sub["week"] <= 145].copy()
    if len(train_history) == 0:
        train_history = sub.copy()
        
    last_week = int(train_history["week"].max())
    category = str(train_history["category"].iloc[-1])
    cuisine = str(train_history["cuisine"].iloc[-1])
    center_type = str(train_history["center_type"].iloc[-1])
    city_code = str(train_history["city_code"].iloc[-1])
    region_code = str(train_history["region_code"].iloc[-1])
    op_area = float(train_history["op_area"].iloc[-1])
    
    # Price defaults (use last observed prices)
    last_checkout_price = float(train_history["checkout_price"].iloc[-1])
    last_base_price = float(train_history["base_price"].iloc[-1])
    
    # Store recent demand history for lag calculations
    recent_orders = list(train_history["num_orders"].tail(8).values)
    
    forecasts = []
    
    for h in range(1, horizon_weeks + 1):
        target_week = last_week + h
        
        # Calculate lag features strictly from prior observations
        lag_1 = float(recent_orders[-1]) if len(recent_orders) >= 1 else float(train_stats["global_mean"])
        lag_2 = float(recent_orders[-2]) if len(recent_orders) >= 2 else lag_1
        lag_3 = float(recent_orders[-3]) if len(recent_orders) >= 3 else lag_2
        lag_4 = float(recent_orders[-4]) if len(recent_orders) >= 4 else lag_3
        
        rolling_4 = recent_orders[-4:]
        rolling_8 = recent_orders[-8:]
        
        rolling_mean_4 = float(np.mean(rolling_4))
        rolling_std_4 = float(np.std(rolling_4))
        rolling_mean_8 = float(np.mean(rolling_8))
        rolling_std_8 = float(np.std(rolling_8))
        rolling_min_4 = float(np.min(rolling_4))
        rolling_max_4 = float(np.max(rolling_4))
        
        # Construct feature row dictionary
        discount_amount = last_base_price - last_checkout_price
        discount_percent = max(discount_amount / max(last_base_price, 1e-5), 0.0)
        price_ratio = last_checkout_price / max(last_base_price, 1e-5)
        
        # Look up target stats from train_stats
        cat_mean = float(train_stats["cat_stats"].loc[train_stats["cat_stats"]["category"] == category, "category_mean_orders"].values[0]) if category in train_stats["cat_stats"]["category"].values else train_stats["global_mean"]
        cuisine_mean = float(train_stats["cuisine_stats"].loc[train_stats["cuisine_stats"]["cuisine"] == cuisine, "cuisine_mean_orders"].values[0]) if cuisine in train_stats["cuisine_stats"]["cuisine"].values else train_stats["global_mean"]
        center_mean = float(train_stats["center_stats"].loc[train_stats["center_stats"]["center_id"] == center_id, "center_mean_orders"].values[0]) if center_id in train_stats["center_stats"]["center_id"].values else train_stats["global_mean"]
        meal_mean = float(train_stats["meal_stats"].loc[train_stats["meal_stats"]["meal_id"] == meal_id, "meal_mean_orders"].values[0]) if meal_id in train_stats["meal_stats"]["meal_id"].values else train_stats["global_mean"]
        
        cm_sub = train_stats["center_meal_stats"][(train_stats["center_meal_stats"]["center_id"] == center_id) & (train_stats["center_meal_stats"]["meal_id"] == meal_id)]
        center_meal_mean = float(cm_sub["center_meal_mean_orders"].values[0]) if len(cm_sub) > 0 else meal_mean

        row_dict = {
            "week": target_week,
            "center_id": center_id,
            "meal_id": meal_id,
            "checkout_price": last_checkout_price,
            "base_price": last_base_price,
            "emailer_for_promotion": 0,
            "homepage_featured": 0,
            "discount_amount": discount_amount,
            "discount_percent": discount_percent,
            "price_ratio": price_ratio,
            "promo_combo": 0,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "lag_4": lag_4,
            "rolling_mean_4": rolling_mean_4,
            "rolling_std_4": rolling_std_4,
            "rolling_mean_8": rolling_mean_8,
            "rolling_std_8": rolling_std_8,
            "rolling_min_4": rolling_min_4,
            "rolling_max_4": rolling_max_4,
            "checkout_price_lag_1": last_checkout_price,
            "price_diff_lag_1": 0.0,
            "price_pct_change": 0.0,
            "category_mean_orders": cat_mean,
            "category_std_orders": 0.0,
            "cuisine_mean_orders": cuisine_mean,
            "cuisine_std_orders": 0.0,
            "center_mean_orders": center_mean,
            "center_std_orders": 0.0,
            "meal_mean_orders": meal_mean,
            "meal_std_orders": 0.0,
            "center_meal_mean_orders": center_meal_mean,
            "center_meal_std_orders": 0.0,
            "category": category,
            "cuisine": cuisine,
            "center_type": center_type,
            "city_code": city_code,
            "region_code": region_code,
            "op_area": op_area
        }
        
        row_df = pd.DataFrame([row_dict])
        row_encoded = encode_categorical_features(row_df).reindex(columns=feature_cols, fill_value=0)
        
        pred_log = model.predict(row_encoded)[0]
        pred_orders = float(np.expm1(pred_log))
        point_forecast = max(round(pred_orders), 0)
        
        # Empirical Holdout Prediction Interval (95%): sqrt(h) scaled validation RMSE
        margin = round(1.96 * val_rmse * np.sqrt(h))
        lower_bound = max(point_forecast - margin, 0)
        upper_bound = point_forecast + margin
        
        forecasts.append({
            "Forecast Week": target_week,
            "Center ID": center_id,
            "Meal ID": meal_id,
            "Category": category,
            "Cuisine": cuisine,
            "Point Forecast Demand": point_forecast,
            "Prediction Interval Lower (95%)": lower_bound,
            "Prediction Interval Upper (95%)": upper_bound,
            "Prior Week Demand (Lag 1)": round(lag_1),
            "Rolling 4W Avg": round(rolling_mean_4, 1)
        })
        
        # Update recent order history recursively with newly forecasted demand
        recent_orders.append(point_forecast)
        
    return pd.DataFrame(forecasts), None
