import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error

def calculate_metrics(y_true, y_pred):
    """
    Calculates MAE, RMSE, RMSLE, WAPE, and R2 metrics.
    Ensures non-negative predictions for RMSLE and demand calculations.
    """
    y_true = np.array(y_true)
    y_pred = np.maximum(np.array(y_pred), 0.0) # non-negative demand constraint
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # RMSLE (log1p handles 0)
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    
    # WAPE: Weighted Absolute Percentage Error = sum(|y - y_pred|) / sum(y) * 100
    sum_true = np.sum(y_true)
    wape = (np.sum(np.abs(y_true - y_pred)) / sum_true * 100.0) if sum_true > 0 else 0.0
    
    # R2 Score
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "RMSLE": round(rmsle, 4),
        "WAPE (%)": round(wape, 2),
        "R2": round(r2, 4)
    }

def get_metric_explanations():
    return {
        "MAE": "Mean Absolute Error: Average absolute difference between forecasted and actual orders in meal units.",
        "RMSE": "Root Mean Squared Error: Penalizes larger forecasting errors more severely, useful for spotting big stockout/surplus risks.",
        "RMSLE": "Root Mean Squared Logarithmic Error: Standard Kaggle evaluation metric. Measures relative ratio error and penalizes under-forecasting more than over-forecasting.",
        "WAPE (%)": "Weighted Absolute Percentage Error: Total absolute error normalized by total actual demand volume. Preferred over MAPE because it avoids division-by-zero or extreme skew on low-volume meals.",
        "R2": "Coefficient of Determination: Proportion of demand variance explained by the ML regression model (1.0 = perfect fit)."
    }
