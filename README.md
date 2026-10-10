# 🥗 FoodWise AI — Smart Food Demand Forecasting & Operational Surplus Prevention System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-11B48E?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![UN SDG 12](https://img.shields.io/badge/UN_SDG-12_Responsible_Consumption-279B48?style=for-the-badge)](https://sdgs.un.org/goals/goal12)
[![UN SDG 2](https://img.shields.io/badge/UN_SDG-2_Zero_Hunger-DDA83A?style=for-the-badge)](https://sdgs.un.org/goals/goal2)

---

## 📌 Problem Statement & Business Motivation

Food-delivery companies, institutional kitchens, and meal-preparation fulfillment centres must accurately estimate demand before preparing meals and purchasing raw ingredients. 
- **Underestimating demand** causes stockouts, unfulfilled customer orders, lost revenue, and poor customer experience.
- **Overestimating demand** leads to excess meal inventory, spoilage, high waste disposal costs, and unnecessary operating expenses.

**FoodWise AI** addresses this operational challenge by providing an end-to-end, machine-learning-driven decision support system that forecasts weekly demand per fulfillment center and meal, explains prediction drivers, and simulates kitchen preparation plans to minimize potential surplus and shortage risks.

> **Sustainability Alignment & Disclosure:** Aligned primarily with **UN SDG 12 (Responsible Consumption and Production)** and secondarily with **UN SDG 2 (Zero Hunger)**. The primary dataset contains meal order counts (`num_orders`), not direct measurements of food waste in kilograms. Spoilage risk and financial exposures are presented as intended operational scenario benefits modeled from user-configured cost parameters.

---

## 📊 Dataset Overview & Limitations

Using the Kaggle **Food Demand Forecasting Dataset**:
- **Dataset File Inventory:**
  - `train.csv`: 456,548 order records across 145 weeks.
  - `test.csv`: 32,573 test order records across 10 future weeks (Weeks 146–155).
  - `meal_info.csv`: Metadata for 51 unique meals (Category, Cuisine).
  - `fulfilment_center_info.csv`: Metadata for 77 fulfillment centres (Center Type, Operational Area, City & Region codes).
- **Target Variable:** `num_orders` (Weekly total orders per center-meal pair). Range: 13 to 24,299 orders; Mean: 261.87 orders.
- **Limitations:** Does not record physical waste quantities or exact calendar dates. Week numbers are used for temporal representation.

---

## ⚙️ Data Preprocessing & Feature Engineering

To prevent **Target Leakage**, time-series lag and rolling features are engineered strictly from prior observations ($week < W$):

1. **Price & Promotional Interaction Features:**
   - `discount_amount` = `base_price` - `checkout_price`
   - `discount_percent` = `discount_amount / base_price`
   - `price_ratio` = `checkout_price / base_price`
   - `promo_combo` = `emailer_for_promotion + 2 * homepage_featured` (0: None, 1: Email, 2: Homepage, 3: Both)
2. **Strict Time-Series Lag Features (per Center-Meal):**
   - Lags: `lag_1`, `lag_2`, `lag_3`, `lag_4`
   - Rolling Window Statistics (derived strictly from `lag_1` prior values): `rolling_mean_4`, `rolling_std_4`, `rolling_mean_8`, `rolling_std_8`, `rolling_min_4`, `rolling_max_4`
3. **Target Encodings:**
   - Historical mean orders per category, cuisine, center, and meal are computed **strictly on the training split** (Weeks 1–135) to prevent validation data leakage.

---

## 🤖 Machine Learning Methodology & Measured Results

Evaluation is conducted using a strict **chronological holdout split**:
- **Training Period:** Weeks 1 to 135 (423,727 historical records)
- **Validation Holdout:** Weeks 136 to 145 (32,821 historical records)
- **Target Transformation:** Models are trained on $\log(1 + y)$ to prevent negative demand predictions and optimize RMSLE. Final predictions are transformed using $\exp(\hat{y}) - 1$ and clipped at $0$.

### 🏆 Validation Leaderboard (Weeks 136 – 145)

| Model Candidate | MAE | RMSE | RMSLE | WAPE (%) | R² | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Naive Lag-1 Baseline** | 79.12 | 136.08 | 0.6708 | 34.81% | 0.7579 | Baseline |
| **Ridge Regression** | 103.13 | 499.83 | 0.5938 | 45.37% | -2.2656 | Evaluated |
| **Random Forest Regressor** | 63.54 | 110.08 | 0.4881 | 27.96% | 0.8416 | Evaluated |
| **HistGradientBoosting Regressor** | **63.12** | **109.93** | 0.4798 | **27.77%** | **0.8420** | Lowest MAE / RMSE / WAPE & Highest R² |
| **XGBoost Regressor (Selected)** | 64.64 | 114.70 | **0.4795** | 28.44% | 0.8280 | **Selected Model (Lowest RMSLE)** |

> **Model Selection Rationale:** `XGBoost Regressor` is selected as the primary production model because it achieves the lowest **RMSLE** (`0.4795`), the official Kaggle log-scale error metric. **HistGradientBoosting Regressor** achieves the lowest absolute metrics: MAE (`63.12`), RMSE (`109.93`), WAPE (`27.77%`), and highest R² (`0.8420`).

---

## 🎯 Forecast Uncertainty & Prediction Intervals

Point forecasts represent the model's expected demand order count. 
**Empirical Prediction Intervals (95%)** are derived from the model's holdout validation RMSE ($114.70$) scaled recursively by the forecast horizon step $h$:

$$\text{Margin}_h = 1.96 \times \text{RMSE}_{\text{val}} \times \sqrt{h}$$

$$\text{Lower}_h = \max(\hat{y}_h - \text{Margin}_h, 0), \quad \text{Upper}_h = \hat{y}_h + \text{Margin}_h$$

---

## 🍳 Surplus & Shortage Prevention Simulator

The operational simulator calculates potential food surplus and shortage stockout risks under user-specified kitchen prep quantities:

$$\text{Recommended Prep} = \text{round}\left(\text{Forecast Demand} \times \left(1 + \frac{\text{Safety Buffer \%}}{100}\right)\right)$$

$$\text{Potential Surplus} = \max(\text{Planned Prep} - \text{Forecast Demand}, 0)$$

$$\text{Potential Shortage} = \max(\text{Forecast Demand} - \text{Planned Prep}, 0)$$

$$\text{Scenario Cost Exposure} = (\text{Surplus Units} \times \text{Cost}_{\text{unsold}}) + (\text{Shortage Units} \times \text{Penalty}_{\text{shortage}})$$

- **Currency Flexibility:** Users can select ₹ INR (Default for Indian business context), $ USD, € EUR, or generic Cost Units.
- **Scenario Disclaimer:** Calculated exposures are interactive scenario calculations based on user inputs, not proven real-world financial savings.

---

## 🤖 AI Operations Copilot Workflow

The application implements a 4-Agent agentic architecture:
1. **Agent 1 — Demand Analyst:** Summarizes demand trends, lag statistics, and percentage changes.
2. **Agent 2 — Kitchen Planning Agent:** Recommends preparation quantities with safety buffers.
3. **Agent 3 — Surplus & Shortage Risk Agent:** Evaluates financial exposure and inventory risks.
4. **Agent 4 — Operations Report Agent:** Synthesizes executive guidance. Optional Gemini LLM integration via `GEMINI_API_KEY` for natural language explanations, with deterministic template fallback when no key is present.

---

## 💻 Local Environment Setup & Execution

### 1. Prerequisites
- Python 3.12
- Git

### 2. Virtual Environment Setup (Windows PowerShell)
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 3. Model Training & Data Preprocessing
```powershell
# Train ML pipeline and generate compact deployment artifacts
python -m src.train
```

### 4. Run Interactive Streamlit Dashboard
```powershell
streamlit run app.py
```
App will open automatically at **`http://localhost:8501`**.

---

## ☁️ Deployment Instructions (Streamlit Community Cloud)

1. Push code repository to GitHub (compact deployment artifacts in `data/processed/*.parquet` and `models/*.joblib` are pre-packaged).
2. Connect repository to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Select Main file path: `app.py`.
4. Ensure Python version is set to **3.12**.
5. (Optional) Set `GEMINI_API_KEY` in Streamlit Cloud Secrets for LLM synthesis.

---

## 📁 Project Structure

```
FoodWiseAI/
├── .streamlit/
│   └── config.toml          # Dark theme dashboard styling
├── data/
│   ├── raw/                 # Raw Kaggle CSV files
│   └── processed/           # Compact deployment parquet datasets
├── models/
│   ├── best_model.joblib    # Trained XGBoost production pipeline
│   └── model_metadata.json  # Validation metrics & feature importance
├── src/
│   ├── data_loader.py       # Data loading & raw dataset inventory
│   ├── preprocessing.py    # Cleaning & engineered price/promo features
│   ├── feature_engineering.py # Leakage-free lag & target encodings
│   ├── evaluate.py          # Metric calculations (MAE, RMSE, RMSLE, WAPE, R2)
│   ├── train.py             # ML pipeline training script
│   ├── forecasting.py       # Recursive multi-week forecasting engine
│   ├── planning.py          # Surplus & shortage scenario simulator
│   └── agents.py            # 4-Agent AI Operations Copilot
├── scratch/                 # Inspection scripts
├── app.py                   # Main Streamlit web application
├── requirements.txt         # Pinned Python dependencies
├── .gitignore               # Deployment git exclusion rules
└── README.md                # Technical documentation
```
