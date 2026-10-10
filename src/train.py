import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor

from src.data_loader import load_merged_data, PROCESSED_DIR
from src.preprocessing import preprocess_dataframe
from src.feature_engineering import create_time_series_features, apply_target_encodings, encode_categorical_features, get_feature_columns
from src.evaluate import calculate_metrics

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

VAL_WEEKS_COUNT = 10 # Last 10 weeks holdout (weeks 136..145)

def train_and_evaluate_all():
    print("Step 1: Loading raw data...")
    merged_train, merged_test, meal_df, center_df = load_merged_data()
    
    print("Step 2: Preprocessing datasets...")
    train_clean = preprocess_dataframe(merged_train)
    test_clean = preprocess_dataframe(merged_test) if merged_test is not None else None
    
    # Combine train + test to generate seamless time-series lag features across all weeks
    if test_clean is not None:
        full_df = pd.concat([train_clean, test_clean], ignore_index=True)
    else:
        full_df = train_clean.copy()
        
    print("Step 3: Creating time-series lag and rolling features...")
    full_df = create_time_series_features(full_df)
    
    # Separate back train and test
    train_feat = full_df[full_df["week"] <= 145].copy()
    test_feat = full_df[full_df["week"] > 145].copy() if test_clean is not None else None
    
    # Define chronological train/val split
    max_train_week = train_feat["week"].max() # 145
    val_start_week = max_train_week - VAL_WEEKS_COUNT + 1 # 136
    
    train_split = train_feat[train_feat["week"] < val_start_week].copy()
    val_split = train_feat[train_feat["week"] >= val_start_week].copy()
    
    print(f"Train split: Weeks {train_split['week'].min()} to {train_split['week'].max()} ({len(train_split)} rows)")
    print(f"Validation split: Weeks {val_split['week'].min()} to {val_split['week'].max()} ({len(val_split)} rows)")
    
    print("Step 4: Applying target encodings fitted strictly on train split...")
    train_split_enc, train_stats = apply_target_encodings(train_split, train_split, train_stats=None)
    val_split_enc, _ = apply_target_encodings(val_split, train_split, train_stats=train_stats)
    full_train_enc, full_train_stats = apply_target_encodings(train_feat, train_feat, train_stats=None)
    
    if test_feat is not None:
        test_feat_enc, _ = apply_target_encodings(test_feat, train_feat, train_stats=full_train_stats)
    
    print("Step 5: One-hot encoding categorical variables...")
    # Get all dummy columns across full train to align features
    full_encoded = encode_categorical_features(full_train_enc)
    feature_cols = get_feature_columns(full_encoded)
    
    # Align feature columns for splits
    X_train_split = encode_categorical_features(train_split_enc).reindex(columns=feature_cols, fill_value=0)
    X_val_split = encode_categorical_features(val_split_enc).reindex(columns=feature_cols, fill_value=0)
    X_full_train = full_encoded.reindex(columns=feature_cols, fill_value=0)
    
    y_train_split = train_split_enc["num_orders"].values
    y_val_split = val_split_enc["num_orders"].values
    y_full_train = full_train_enc["num_orders"].values
    
    # Target log transformation log1p
    y_train_split_log = np.log1p(y_train_split)
    y_val_split_log = np.log1p(y_val_split)
    y_full_train_log = np.log1p(y_full_train)
    
    # Handle any remaining NaNs in features
    X_train_split = X_train_split.fillna(0)
    X_val_split = X_val_split.fillna(0)
    X_full_train = X_full_train.fillna(0)
    
    print(f"Number of feature columns: {len(feature_cols)}")
    
    # Define Candidate Models
    models = {
        "Naive Lag-1 Baseline": None,
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=60, max_depth=12, n_jobs=-1, random_state=42),
        "HistGradientBoosting Regressor": HistGradientBoostingRegressor(max_iter=100, max_depth=8, random_state=42),
        "XGBoost Regressor": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.08, n_jobs=-1, random_state=42)
    }
    
    results = {}
    fitted_val_models = {}
    val_predictions = {}
    
    # Evaluate Baseline
    naive_preds = val_split_enc["lag_1"].fillna(val_split_enc["rolling_mean_4"]).fillna(100).values
    results["Naive Lag-1 Baseline"] = calculate_metrics(y_val_split, naive_preds)
    val_predictions["Naive Lag-1 Baseline"] = naive_preds
    
    for name, model in models.items():
        if model is None:
            continue
        print(f"Training candidate model: {name}...")
        model.fit(X_train_split, y_train_split_log)
        preds_log = model.predict(X_val_split)
        preds = np.expm1(preds_log)
        preds = np.maximum(preds, 0.0)
        
        metrics = calculate_metrics(y_val_split, preds)
        results[name] = metrics
        fitted_val_models[name] = model
        val_predictions[name] = preds
        print(f" -> {name} Validation Metrics: {metrics}")
        
    # Select best model based on lowest RMSLE
    best_model_name = min(results.keys(), key=lambda k: results[k]["RMSLE"])
    best_val_rmse = float(results[best_model_name]["RMSE"])
    print(f"\nSELECTED BEST MODEL: {best_model_name} (RMSLE: {results[best_model_name]['RMSLE']}, MAE: {results[best_model_name]['MAE']}, Val RMSE: {best_val_rmse})")
    
    # Retrain best model on FULL dataset (weeks 1..145) for production forecasting
    print(f"Retraining {best_model_name} on full dataset (weeks 1..145)...")
    if best_model_name == "Naive Lag-1 Baseline":
        final_model = models["XGBoost Regressor"]
        final_model.fit(X_full_train, y_full_train_log)
    else:
        final_model = models[best_model_name]
        final_model.fit(X_full_train, y_full_train_log)
        
    # Extract feature importance if available
    feature_importance = {}
    if hasattr(final_model, "feature_importances_"):
        importances = final_model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        feature_importance = {feature_cols[i]: float(importances[i]) for i in sorted_idx[:25]}
    elif hasattr(final_model, "coef_"):
        coefs = np.abs(final_model.coef_)
        sorted_idx = np.argsort(coefs)[::-1]
        feature_importance = {feature_cols[i]: float(coefs[i]) for i in sorted_idx[:25]}

    # Save artifacts
    model_artifact_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump({
        "model": final_model,
        "best_model_name": best_model_name,
        "feature_cols": feature_cols,
        "full_train_stats": full_train_stats,
        "val_rmse": best_val_rmse
    }, model_artifact_path)
    
    # Save metadata summary
    metadata = {
        "best_model_name": best_model_name,
        "best_val_rmse": best_val_rmse,
        "selection_criterion": "Lowest RMSLE on validation holdout split",
        "results": results,
        "val_weeks_range": f"Weeks {val_start_week} to {max_train_week}",
        "train_weeks_range": f"Weeks 1 to {val_start_week - 1}",
        "feature_importance": feature_importance,
        "feature_count": len(feature_cols),
        "total_train_records": len(train_feat),
        "unique_centers": int(train_feat["center_id"].nunique()),
        "unique_meals": int(train_feat["meal_id"].nunique())
    }
    
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save compact processed parquet datasets
    print("Step 7: Saving compact processed datasets for deployment...")
    val_export = val_split[["week", "center_id", "meal_id", "category", "cuisine", "num_orders"]].copy()
    for name in results.keys():
        val_export[f"pred_{name}"] = val_predictions[name]
    val_export.to_parquet(os.path.join(PROCESSED_DIR, "validation_predictions.parquet"))
    
    # Store compact aggregated demand trend for EDA
    weekly_trend = train_feat.groupby("week")["num_orders"].sum().reset_index()
    weekly_trend.to_parquet(os.path.join(PROCESSED_DIR, "weekly_demand_trend.parquet"))
    
    category_trend = train_feat.groupby(["category", "cuisine"])["num_orders"].agg(["sum", "mean", "count"]).reset_index()
    category_trend.to_parquet(os.path.join(PROCESSED_DIR, "category_summary.parquet"))
    
    center_trend = train_feat.groupby(["center_id", "center_type", "op_area"])["num_orders"].agg(["sum", "mean", "count"]).reset_index()
    center_trend.to_parquet(os.path.join(PROCESSED_DIR, "center_summary.parquet"))
    
    # Store processed full features compact subset (last 20 weeks + test weeks) for forecasting
    compact_features = full_df[full_df["week"] >= 120].copy()
    compact_features.to_parquet(os.path.join(PROCESSED_DIR, "compact_features.parquet"))

    print("Training and artifact export complete successfully!")
    return metadata

if __name__ == "__main__":
    train_and_evaluate_all()
