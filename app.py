import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import (
    load_merged_data, inspect_dataset_inventory, PROCESSED_DIR
)
from src.forecasting import generate_multi_week_forecast, load_forecasting_artifacts
from src.planning import simulate_surplus_and_shortage, generate_plan_comparison
from src.agents import run_operations_copilot
from src.evaluate import get_metric_explanations

# Streamlit Page Config
st.set_page_config(
    page_title="FoodWise AI — Smart Demand & Surplus Planning",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI Styling
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Container */
    .header-box {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .header-title {
        color: #10B981;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 6px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: #10B981;
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    
    .metric-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 6px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1E293B;
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 6px;
        color: #94A3B8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #10B981 !important;
        color: #FFFFFF !important;
    }

    /* Callout Box */
    .sdg-box {
        background: rgba(16, 185, 129, 0.08);
        border-left: 4px solid #10B981;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 20px;
        font-size: 0.95rem;
        color: #CBD5E1;
    }
</style>
""", unsafe_allow_html=True)

# Helper cached data loaders
@st.cache_data
def load_processed_summaries():
    trend_path = os.path.join(PROCESSED_DIR, "weekly_demand_trend.parquet")
    cat_path = os.path.join(PROCESSED_DIR, "category_summary.parquet")
    center_path = os.path.join(PROCESSED_DIR, "center_summary.parquet")
    val_path = os.path.join(PROCESSED_DIR, "validation_predictions.parquet")
    
    trend_df = pd.read_parquet(trend_path) if os.path.exists(trend_path) else None
    cat_df = pd.read_parquet(cat_path) if os.path.exists(cat_path) else None
    center_df = pd.read_parquet(center_path) if os.path.exists(center_path) else None
    val_df = pd.read_parquet(val_path) if os.path.exists(val_path) else None
    
    return trend_df, cat_df, center_df, val_df

@st.cache_data
def load_full_merged_sample():
    merged_train, merged_test, meal_df, center_df = load_merged_data()
    return merged_train, merged_test, meal_df, center_df

@st.cache_resource
def load_model_metadata():
    meta_path = os.path.join(os.path.dirname(__file__), "models", "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return None

# Load global metadata & cached data
metadata = load_model_metadata()
trend_df, cat_df, center_df, val_df = load_processed_summaries()

# Sidebar Navigation
st.sidebar.markdown(
    """
    <div style="
        font-size: 52px;
        line-height: 1.3;
        margin-bottom: 8px;
    ">🥗</div>
    """,
    unsafe_allow_html=True
)
st.sidebar.title("FoodWise AI")
st.sidebar.markdown("**Demand Forecasting & Surplus Prevention**")

nav_choice = st.sidebar.radio(
    "Navigate System",
    [
        "📊 A. Overview",
        "📈 B. Demand Analytics",
        "🎯 C. Demand Forecast",
        "🍳 D. Kitchen Planning Simulator",
        "🧠 E. AI Operations Copilot",
        "⚡ F. Model Performance",
        "🔍 G. Data Quality & Inventory"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### UN SDG Alignment")
st.sidebar.markdown("🌱 **Primary:** SDG 12 — Responsible Consumption & Production")
st.sidebar.markdown("🍲 **Secondary:** SDG 2 — Zero Hunger")
st.sidebar.caption("Notice: Waste prevention is presented as an intended operational benefit based on user scenario modeling.")

# -------------------------------------------------------------------
# A. OVERVIEW
# -------------------------------------------------------------------
if nav_choice == "📊 A. Overview":
    st.markdown("""
    <div class="header-box">
        <h1 class="header-title">FoodWise AI</h1>
        <div class="header-subtitle">Smart Food Demand Forecasting & Operational Surplus Prevention System</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sdg-box">
        <b>UN Sustainability Goal Alignment:</b> FoodWise AI helps institutional kitchens and fulfillment centres accurately predict weekly meal demand, minimizing stockouts while assisting operators in preventing excess inventory and food waste.
    </div>
    """, unsafe_allow_html=True)

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">456,548</div>
            <div class="metric-label">Training Records</div>
            <div class="metric-badge">Weeks 1-145</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">77</div>
            <div class="metric-label">Fulfillment Centres</div>
            <div class="metric-badge">3 Center Types</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">51</div>
            <div class="metric-label">Unique Meals</div>
            <div class="metric-badge">14 Categories</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        best_name = metadata.get("best_model_name", "XGBoost Regressor") if metadata else "XGBoost Regressor"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size: 1.3rem;">{best_name}</div>
            <div class="metric-label">Selected Model</div>
            <div class="metric-badge">Lowest RMSLE</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        mae_val = metadata["results"][best_name]["MAE"] if metadata and best_name in metadata["results"] else "64.64"
        rmsle_val = metadata["results"][best_name]["RMSLE"] if metadata and best_name in metadata["results"] else "0.4795"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{mae_val}</div>
            <div class="metric-label">Validation MAE</div>
            <div class="metric-badge">RMSLE: {rmsle_val}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Overview Visualizations
    col_left, col_right = st.columns([6, 4])
    with col_left:
        st.subheader("📈 Overall Weekly Demand Trend (Weeks 1 – 145)")
        if trend_df is not None:
            fig_trend = px.line(
                trend_df, x="week", y="num_orders",
                labels={"week": "Week Number", "num_orders": "Total Orders"},
                template="plotly_dark", color_discrete_sequence=["#10B981"]
            )
            fig_trend.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=340)
            st.plotly_chart(fig_trend, use_container_width=True)
    
    with col_right:
        st.subheader("🍲 Demand Share by Category")
        if cat_df is not None:
            cat_agg = cat_df.groupby("category")["sum"].sum().reset_index()
            fig_pie = px.pie(
                cat_agg, names="category", values="sum",
                hole=0.4, template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=340)
            st.plotly_chart(fig_pie, use_container_width=True)

# -------------------------------------------------------------------
# B. DEMAND ANALYTICS
# -------------------------------------------------------------------
elif nav_choice == "📈 B. Demand Analytics":
    st.title("📈 Demand Analytics & Exploratory Data Analysis")
    st.markdown("Interactively explore demand patterns across categories, cuisines, fulfilment centres, prices, and promotions.")

    merged_train, _, meal_df, center_df = load_full_merged_sample()

    # Interactive Filters
    st.subheader("🔍 Filters")
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        selected_cat = st.multiselect("Meal Category", options=sorted(merged_train["category"].unique()), default=sorted(merged_train["category"].unique())[:3])
    with f_col2:
        selected_cuisine = st.multiselect("Cuisine", options=sorted(merged_train["cuisine"].unique()), default=sorted(merged_train["cuisine"].unique()))
    with f_col3:
        week_range = st.slider("Week Range", int(merged_train["week"].min()), int(merged_train["week"].max()), (1, 145))

    # Filter data
    filtered_df = merged_train[
        (merged_train["category"].isin(selected_cat if selected_cat else merged_train["category"].unique())) &
        (merged_train["cuisine"].isin(selected_cuisine if selected_cuisine else merged_train["cuisine"].unique())) &
        (merged_train["week"].between(week_range[0], week_range[1]))
    ]

    st.markdown(f"**Filtered Records:** `{len(filtered_df):,}` orders across selected parameters.")

    tab1, tab2, tab3 = st.tabs(["🏷️ Category & Cuisine", "🏢 Fulfillment Centres", "📣 Price & Promotions"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig_cat = px.box(
                filtered_df, x="category", y="num_orders", color="category",
                title="Order Distribution by Category", template="plotly_dark",
                log_y=True
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        with c2:
            fig_cuis = px.bar(
                filtered_df.groupby("cuisine")["num_orders"].mean().reset_index(),
                x="cuisine", y="num_orders", color="cuisine",
                title="Average Weekly Orders per Cuisine", template="plotly_dark"
            )
            st.plotly_chart(fig_cuis, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig_cent = px.box(
                filtered_df, x="center_type", y="num_orders", color="center_type",
                title="Demand by Center Type", template="plotly_dark", log_y=True
            )
            st.plotly_chart(fig_cent, use_container_width=True)
        with c2:
            fig_op = px.scatter(
                filtered_df.groupby("center_id").agg({"op_area": "first", "num_orders": "mean"}).reset_index(),
                x="op_area", y="num_orders", size="num_orders",
                title="Operational Area vs Average Demand", template="plotly_dark",
                labels={"op_area": "Operational Area (sq km)", "num_orders": "Mean Orders"}
            )
            st.plotly_chart(fig_op, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            promo_summary = filtered_df.groupby(["emailer_for_promotion", "homepage_featured"])["num_orders"].mean().reset_index()
            promo_summary["Campaign"] = promo_summary.apply(
                lambda r: "Both Promos" if r["emailer_for_promotion"]==1 and r["homepage_featured"]==1
                else ("Email Only" if r["emailer_for_promotion"]==1
                else ("Homepage Only" if r["homepage_featured"]==1 else "No Promo")), axis=1
            )
            fig_promo = px.bar(
                promo_summary, x="Campaign", y="num_orders", color="Campaign",
                title="Impact of Promotional Campaigns on Mean Demand", template="plotly_dark"
            )
            st.plotly_chart(fig_promo, use_container_width=True)
        with c2:
            fig_price = px.scatter(
                filtered_df.sample(min(5000, len(filtered_df)), random_state=42),
                x="checkout_price", y="num_orders", color="category",
                title="Checkout Price vs Demand (Sampled 5k)", template="plotly_dark",
                opacity=0.6, log_y=True
            )
            st.plotly_chart(fig_price, use_container_width=True)

# -------------------------------------------------------------------
# C. DEMAND FORECAST
# -------------------------------------------------------------------
elif nav_choice == "🎯 C. Demand Forecast":
    st.title("🎯 ML Demand Forecast Generator")
    st.markdown("Generate recursive multi-week demand forecasts with empirical prediction intervals (95% holdout error bounds).")

    merged_train, _, meal_df, center_df = load_full_merged_sample()

    fc_col1, fc_col2, fc_col3 = st.columns(3)
    with fc_col1:
        center_id = st.selectbox("Select Fulfillment Center ID", options=sorted(merged_train["center_id"].unique()), index=0)
    with fc_col2:
        avail_meals = sorted(merged_train[merged_train["center_id"] == center_id]["meal_id"].unique())
        meal_id = st.selectbox("Select Meal ID", options=avail_meals, index=0)
    with fc_col3:
        horizon_weeks = st.slider("Forecast Horizon (Weeks)", min_value=1, max_value=10, value=4)

    meal_meta = meal_df[meal_df["meal_id"] == meal_id].iloc[0]
    st.info(f"📍 **Selected Combination:** Center {center_id} | Meal {meal_id} ({meal_meta['category']} - {meal_meta['cuisine']} Cuisine)")

    if st.button("🚀 Generate Multi-Week Forecast", type="primary"):
        with st.spinner("Executing ML forecasting pipeline..."):
            forecast_df, err = generate_multi_week_forecast(center_id=center_id, meal_id=meal_id, horizon_weeks=horizon_weeks)
            if err:
                st.error(err)
            else:
                st.success("✅ Multi-Week Forecast Generated Successfully!")
                
                col_table, col_chart = st.columns([5, 5])
                with col_table:
                    st.subheader("📋 Forecast Demand Table")
                    display_cols = ["Forecast Week", "Point Forecast Demand", "Prediction Interval Lower (95%)", "Prediction Interval Upper (95%)", "Prior Week Demand (Lag 1)", "Rolling 4W Avg"]
                    st.dataframe(
                        forecast_df[display_cols],
                        use_container_width=True
                    )
                    st.caption("Note: Point Forecast Demand represents the model's expected order count. Prediction intervals (95%) reflect empirical holdout residual uncertainty scaled recursively over multi-week steps.")
                
                with col_chart:
                    st.subheader("📉 Forecast Trend & Empirical Prediction Interval")
                    fig_fc = go.Figure()
                    
                    # Upper & Lower bounds shadow
                    fig_fc.add_trace(go.Scatter(
                        x=forecast_df["Forecast Week"], y=forecast_df["Prediction Interval Upper (95%)"],
                        mode='lines', line=dict(width=0), showlegend=False
                    ))
                    fig_fc.add_trace(go.Scatter(
                        x=forecast_df["Forecast Week"], y=forecast_df["Prediction Interval Lower (95%)"],
                        mode='lines', line=dict(width=0), fill='tonexty',
                        fillcolor='rgba(16, 185, 129, 0.2)', name='Empirical Prediction Interval (95%)'
                    ))
                    
                    # Point Forecast
                    fig_fc.add_trace(go.Scatter(
                        x=forecast_df["Forecast Week"], y=forecast_df["Point Forecast Demand"],
                        mode='lines+markers', line=dict(color='#10B981', width=3),
                        name='Point Forecast Demand'
                    ))
                    
                    fig_fc.update_layout(
                        template="plotly_dark",
                        xaxis_title="Future Week Number",
                        yaxis_title="Demand (Meal Units)",
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    st.plotly_chart(fig_fc, use_container_width=True)

# -------------------------------------------------------------------
# D. KITCHEN PLANNING SIMULATOR
# -------------------------------------------------------------------
elif nav_choice == "🍳 D. Kitchen Planning Simulator":
    st.title("🍳 Surplus & Shortage Prevention Simulator")
    st.markdown("Simulate preparation plans against ML demand forecasts to evaluate potential surplus food, shortage stockouts, and scenario cost exposures.")

    sim_col1, sim_col2 = st.columns([4, 6])
    
    with sim_col1:
        st.subheader("⚙️ Simulation Controls")
        currency_sym = st.selectbox("Currency / Unit Label", options=["₹", "$", "€", "Units"], index=0)
        forecast_input = st.number_input("ML Forecasted Demand (Units)", min_value=1, value=250, step=10)
        planned_prep = st.number_input("Planned Preparation Quantity (Units)", min_value=1, value=280, step=10)
        safety_buffer_pct = st.slider("Recommended Safety Buffer (%)", min_value=0.0, max_value=30.0, value=10.0, step=1.0)
        
        st.markdown("---")
        st.markdown(f"##### 💵 Scenario Cost Parameters ({currency_sym})")
        cost_unsold = st.number_input(f"Cost per Unsold Surplus Unit ({currency_sym})", min_value=0.0, value=50.0 if currency_sym=="₹" else 3.50, step=5.0 if currency_sym=="₹" else 0.50)
        cost_shortage = st.number_input(f"Penalty per Unfulfilled Order ({currency_sym})", min_value=0.0, value=120.0 if currency_sym=="₹" else 8.00, step=10.0 if currency_sym=="₹" else 0.50)

    # Run simulation calculation
    sim_res = simulate_surplus_and_shortage(
        forecast_demand=forecast_input,
        planned_quantity=planned_prep,
        service_buffer_pct=safety_buffer_pct,
        cost_per_unsold=cost_unsold,
        cost_per_shortage=cost_shortage,
        currency_symbol=currency_sym
    )

    with sim_col2:
        st.subheader("📊 Scenario Outcome Cards")
        
        sc_col1, sc_col2, sc_col3 = st.columns(3)
        with sc_col1:
            st.metric("ML Forecast Demand", f"{sim_res['forecast_demand']} units")
        with sc_col2:
            st.metric("Recommended Prep", f"{sim_res['recommended_quantity']} units", f"+{safety_buffer_pct}% buffer")
        with sc_col3:
            st.metric("Planned Prep", f"{sim_res['planned_quantity']} units")

        st.markdown("<br>", unsafe_allow_html=True)
        
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            surplus_color = "#EF4444" if sim_res["potential_surplus_units"] > 0 else "#10B981"
            st.markdown(f"""
            <div class="metric-card" style="border-color: {surplus_color};">
                <div class="metric-value" style="color: {surplus_color};">{sim_res['potential_surplus_units']}</div>
                <div class="metric-label">Potential Surplus Units</div>
                <div class="metric-badge">Spoilage Exposure: {currency_sym}{sim_res['surplus_cost_exposure']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            shortage_color = "#F59E0B" if sim_res["potential_shortage_units"] > 0 else "#10B981"
            st.markdown(f"""
            <div class="metric-card" style="border-color: {shortage_color};">
                <div class="metric-value" style="color: {shortage_color};">{sim_res['potential_shortage_units']}</div>
                <div class="metric-label">Potential Shortage Units</div>
                <div class="metric-badge">Stockout Exposure: {currency_sym}{sim_res['shortage_cost_exposure']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{currency_sym}{sim_res['total_cost_exposure']}</div>
                <div class="metric-label">Total Scenario Exposure</div>
                <div class="metric-badge">User Scenario Calculation</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Alternative Plan Comparison Grid")
    comp_df = generate_plan_comparison(
        forecast_demand=forecast_input,
        current_plan=planned_prep,
        service_buffer_pct=safety_buffer_pct,
        cost_per_unsold=cost_unsold,
        cost_per_shortage=cost_shortage,
        currency_symbol=currency_sym
    )
    st.dataframe(comp_df, use_container_width=True)
    st.caption("Disclaimer: Estimated scenario exposures are transparent user-parameter calculations for operational planning and do not represent proven real-world financial savings or physical waste measurements.")

# -------------------------------------------------------------------
# E. AI OPERATIONS COPILOT
# -------------------------------------------------------------------
elif nav_choice == "🤖 E. AI Operations Copilot":
    st.title(" AI Operations Copilot")
    st.markdown("Multi-agent operational decision support powered by ML forecasts, risk simulation, and optional Gemini LLM synthesis.")

    merged_train, _, _, _ = load_full_merged_sample()

    cp_col1, cp_col2, cp_col3, cp_col4 = st.columns(4)
    with cp_col1:
        center_id = st.selectbox("Fulfillment Center ID", options=sorted(merged_train["center_id"].unique()), index=0)
    with cp_col2:
        avail_meals = sorted(merged_train[merged_train["center_id"] == center_id]["meal_id"].unique())
        meal_id = st.selectbox("Meal ID", options=avail_meals, index=0)
    with cp_col3:
        planned_prep = st.number_input("Kitchen Planned Prep Quantity", min_value=10, value=250, step=10)
    with cp_col4:
        currency_sym = st.selectbox("Currency Symbol", options=["₹", "$", "€", "Units"], index=0)

    if st.button("⚡ Run AI Operations Agents", type="primary"):
        with st.spinner("Orchestrating 4-Agent Operational Workflow..."):
            copilot_out = run_operations_copilot(
                center_id=center_id,
                meal_id=meal_id,
                horizon_weeks=4,
                planned_prep=planned_prep,
                service_buffer_pct=10.0,
                cost_per_unsold=50.0 if currency_sym=="₹" else 3.50,
                cost_per_shortage=120.0 if currency_sym=="₹" else 8.00,
                currency_symbol=currency_sym
            )
            
            if "error" in copilot_out:
                st.error(copilot_out["error"])
            else:
                st.success("✅ Operational Copilot Analysis Complete!")
                
                # Agent Output Tabs
                a_tab1, a_tab2, a_tab3, a_tab4 = st.tabs([
                    "📊 Agent 1: Demand Analyst",
                    "🍳 Agent 2: Kitchen Planner",
                    "⚠️ Agent 3: Surplus & Risk Analyst",
                    "📝 Agent 4: Operations Summary"
                ])
                
                with a_tab1:
                    st.json(copilot_out["agent1_demand_analyst"])
                with a_tab2:
                    st.json(copilot_out["agent2_kitchen_planner"])
                with a_tab3:
                    st.json(copilot_out["agent3_risk_analyst"])
                with a_tab4:
                    st.markdown("### 📜 Executive Operational Summary")
                    st.info(copilot_out["agent4_operations_summary"])
                    
                    if copilot_out["llm_explanation"]:
                        st.markdown("### 🤖 Gemini Natural Language Synthesis")
                        st.write(copilot_out["llm_explanation"])
                    else:
                        st.caption("ℹ️ Running in deterministic mode (No GEMINI_API_KEY detected). Deterministic operational template displayed above.")

# -------------------------------------------------------------------
# F. MODEL PERFORMANCE
# -------------------------------------------------------------------
elif nav_choice == "⚡ F. Model Performance":
    st.title("⚡ Model Performance & Validation Analysis")
    st.markdown("Chronological holdout evaluation (Weeks 136 – 145) comparing baseline and advanced machine learning regressors.")

    if metadata and "results" in metadata:
        results_df = pd.DataFrame(metadata["results"]).T
        results_df = results_df.reset_index().rename(columns={"index": "Candidate Model"})
        
        st.subheader("🏆 Candidate Model Benchmark Leaderboard")
        st.dataframe(results_df, use_container_width=True)

        st.info("💡 **Model Selection Rationale:** **XGBoost Regressor** is selected as the production model because it achieves the lowest **RMSLE** (0.4795). **HistGradientBoosting Regressor** achieves the lowest MAE (63.12), lowest RMSE (109.93), lowest WAPE (27.77%), and highest R² (0.8420).")

        st.markdown("<br>", unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig_mae = px.bar(
                results_df, x="Candidate Model", y="MAE", color="Candidate Model",
                title="MAE Comparison (Lower is Better)", template="plotly_dark"
            )
            st.plotly_chart(fig_mae, use_container_width=True)
        with col_m2:
            fig_rmsle = px.bar(
                results_df, x="Candidate Model", y="RMSLE", color="Candidate Model",
                title="RMSLE Comparison (Lower is Better)", template="plotly_dark"
            )
            st.plotly_chart(fig_rmsle, use_container_width=True)

        # Scatter Plot & Residual Analysis if validation parquet exists
        if val_df is not None:
            st.markdown("---")
            st.subheader("🎯 Actual vs Predicted Demand Scatter & Residuals")
            
            best_col = f"pred_{metadata['best_model_name']}" if f"pred_{metadata['best_model_name']}" in val_df.columns else "pred_XGBoost Regressor"
            
            sc1, sc2 = st.columns(2)
            with sc1:
                fig_scat = px.scatter(
                    val_df.sample(min(3000, len(val_df)), random_state=42),
                    x="num_orders", y=best_col, color="category",
                    title="Actual vs Predicted Orders (Validation Holdout)",
                    labels={"num_orders": "Actual Orders", best_col: "Predicted Orders"},
                    template="plotly_dark", opacity=0.6
                )
                fig_scat.add_trace(go.Scatter(x=[0, 3000], y=[0, 3000], mode='lines', line=dict(color='red', dash='dash'), name='Ideal 1:1 Line'))
                st.plotly_chart(fig_scat, use_container_width=True)
                
            with sc2:
                val_df["residual"] = val_df["num_orders"] - val_df[best_col]
                fig_res = px.histogram(
                    val_df, x="residual", nbins=50, title="Residual Distribution (Actual - Predicted)",
                    template="plotly_dark", color_discrete_sequence=["#10B981"]
                )
                st.plotly_chart(fig_res, use_container_width=True)

        # Feature Importance
        if "feature_importance" in metadata and metadata["feature_importance"]:
            st.markdown("---")
            st.subheader("🔑 Top Feature Importances")
            fi_df = pd.DataFrame(list(metadata["feature_importance"].items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=True)
            fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h", title="Top 25 Drivers of Demand Prediction", template="plotly_dark", color_discrete_sequence=["#10B981"])
            st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("---")
        st.subheader("📚 Metric Explanations & Selection Rationale")
        exps = get_metric_explanations()
        for m_name, m_desc in exps.items():
            st.markdown(f"• **{m_name}:** {m_desc}")

# -------------------------------------------------------------------
# G. DATA QUALITY & INVENTORY
# -------------------------------------------------------------------
elif nav_choice == "🔍 G. Data Quality & Inventory":
    st.title("🔍 Data Quality, Schema & Inventory")
    st.markdown("Audit raw dataset files, schemas, missing values, preprocessing actions, and target leakage prevention measures.")

    inv_df = inspect_dataset_inventory()
    st.subheader("📦 Raw Dataset Inventory")
    st.dataframe(inv_df, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🛡️ Data Processing & Leakage Prevention Policy")
    st.markdown("""
    1. **Chronological Splitting:** Model evaluation uses a strict chronological holdout of the final 10 historical weeks (Weeks 136–145). Random K-Fold cross-validation is avoided to prevent temporal data leakage.
    2. **Lag Feature Integrity:** Lags (`lag_1`, `lag_2`, etc.) and rolling statistics (`rolling_mean_4`, `rolling_std_4`) are calculated strictly on observations prior to week $W$. Future target observations are never used.
    3. **Target Encodings:** Mean historical demand statistics per category, cuisine, and center are fitted exclusively on the training split (Weeks 1–135) and mapped onto validation and test splits.
    4. **Non-Negative Predictions:** Predictions are trained on $\\log(1 + y)$ and inverse-transformed via $\\exp(\\hat{y}) - 1$, clipped at 0 to enforce non-negative meal quantities.
    5. **Dataset Limitation Disclosure:** The primary Kaggle dataset contains meal order counts (`num_orders`), not physical food waste measurements in kilograms. Potential surplus and spoilage risks are operational scenario calculations derived from user-configured cost and buffer parameters.
    """)
