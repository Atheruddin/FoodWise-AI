import os
import json
import pandas as pd
import numpy as np

from src.forecasting import generate_multi_week_forecast
from src.planning import simulate_surplus_and_shortage, generate_plan_comparison

# Optional Gemini LLM import
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

def check_gemini_api_key():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return api_key

def run_demand_analyst_agent(forecast_df, center_id, meal_id):
    """
    Agent 1 — Demand Analyst: Analyzes point forecast demand trends, lag values, and variability.
    """
    demand_col = "Point Forecast Demand" if "Point Forecast Demand" in forecast_df.columns else "Forecasted Demand"
    total_forecast = int(forecast_df[demand_col].sum())
    avg_forecast = float(forecast_df[demand_col].mean())
    max_forecast = int(forecast_df[demand_col].max())
    min_forecast = int(forecast_df[demand_col].min())
    lag_1 = float(forecast_df["Prior Week Demand (Lag 1)"].iloc[0])
    category = str(forecast_df["Category"].iloc[0])
    cuisine = str(forecast_df["Cuisine"].iloc[0])
    
    pct_change = round(((avg_forecast - lag_1) / max(lag_1, 1.0)) * 100.0, 1)
    trend_direction = "an increase" if pct_change > 0 else ("a decrease" if pct_change < 0 else "stable demand")
    
    summary = {
        "agent": "Agent 1 — Demand Analyst",
        "center_id": center_id,
        "meal_id": meal_id,
        "category": category,
        "cuisine": cuisine,
        "total_forecast_horizon_units": total_forecast,
        "avg_weekly_demand": round(avg_forecast, 1),
        "min_weekly_demand": min_forecast,
        "max_weekly_demand": max_forecast,
        "prior_week_lag1": round(lag_1),
        "expected_trend_pct": pct_change,
        "trend_description": f"Forecast indicates {trend_direction} of {abs(pct_change)}% compared to prior week lag-1 baseline ({round(lag_1)} units)."
    }
    return summary

def run_kitchen_planning_agent(forecast_demand, service_buffer_pct):
    """
    Agent 2 — Kitchen Planning Agent: Translates demand predictions into prep targets with safety buffer.
    """
    recommended_prep = round(forecast_demand * (1.0 + service_buffer_pct / 100.0))
    buffer_units = recommended_prep - round(forecast_demand)
    
    summary = {
        "agent": "Agent 2 — Kitchen Planning Agent",
        "forecast_demand": round(forecast_demand),
        "service_buffer_pct": service_buffer_pct,
        "recommended_prep_quantity": recommended_prep,
        "safety_buffer_units": buffer_units,
        "planning_guidance": f"Prepare {recommended_prep} units (includes a {service_buffer_pct}% buffer of +{buffer_units} units above predicted point forecast of {round(forecast_demand)})."
    }
    return summary

def run_risk_agent(forecast_demand, planned_quantity, cost_per_unsold, cost_per_shortage, currency_symbol="₹"):
    """
    Agent 3 — Surplus & Shortage Risk Agent: Evaluates scenario risks and financial exposure.
    """
    sim = simulate_surplus_and_shortage(
        forecast_demand=forecast_demand,
        planned_quantity=planned_quantity,
        service_buffer_pct=0.0,
        cost_per_unsold=cost_per_unsold,
        cost_per_shortage=cost_per_shortage,
        currency_symbol=currency_symbol
    )
    
    risk_level = "Balanced Risk"
    if sim["potential_surplus_units"] > 0 and sim["potential_shortage_units"] == 0:
        risk_level = "High Spoilage Risk (Over-Preparation)"
    elif sim["potential_shortage_units"] > 0 and sim["potential_surplus_units"] == 0:
        risk_level = "High Revenue/Stockout Risk (Under-Preparation)"

    summary = {
        "agent": "Agent 3 — Surplus & Shortage Risk Agent",
        "risk_level": risk_level,
        "potential_surplus_units": sim["potential_surplus_units"],
        "potential_shortage_units": sim["potential_shortage_units"],
        "surplus_cost_exposure": sim["surplus_cost_exposure"],
        "shortage_cost_exposure": sim["shortage_cost_exposure"],
        "total_scenario_exposure": sim["total_cost_exposure"],
        "currency_symbol": currency_symbol
    }
    return summary

def run_operations_copilot(center_id, meal_id, horizon_weeks, planned_prep, service_buffer_pct, cost_per_unsold, cost_per_shortage, currency_symbol="₹"):
    """
    Orchestrates the 4-Agent Operational Copilot Workflow.
    Fuses deterministic tool outputs with optional Gemini natural-language synthesis.
    """
    forecast_df, err = generate_multi_week_forecast(center_id, meal_id, horizon_weeks=horizon_weeks)
    if err or forecast_df is None:
        return {"error": err or "Failed to generate forecast for agent copilot."}
        
    demand_col = "Point Forecast Demand" if "Point Forecast Demand" in forecast_df.columns else "Forecasted Demand"
    w1_forecast = float(forecast_df[demand_col].iloc[0])
    
    agent1_out = run_demand_analyst_agent(forecast_df, center_id, meal_id)
    agent2_out = run_kitchen_planning_agent(w1_forecast, service_buffer_pct)
    agent3_out = run_risk_agent(w1_forecast, planned_prep, cost_per_unsold, cost_per_shortage, currency_symbol=currency_symbol)
    
    # Combined deterministic operational summary (Agent 4)
    summary_text = (
        f"OPERATIONAL EXECUTIVE SUMMARY (Center {center_id} | Meal {meal_id} - {agent1_out['category']})\n"
        f"• Demand Forecast: Week 1 point forecast is {round(w1_forecast)} units ({agent1_out['expected_trend_pct']}% change vs lag-1 baseline).\n"
        f"• Kitchen Planning: Recommended preparation is {agent2_out['recommended_prep_quantity']} units with a {service_buffer_pct}% safety buffer.\n"
        f"• Risk Assessment: Current planned quantity ({planned_prep} units) results in {agent3_out['potential_surplus_units']} potential surplus units and {agent3_out['potential_shortage_units']} potential shortage units.\n"
        f"• Scenario Cost Exposure: Estimated total scenario exposure is {currency_symbol}{agent3_out['total_scenario_exposure']:.2f} ({currency_symbol}{agent3_out['surplus_cost_exposure']:.2f} surplus / {currency_symbol}{agent3_out['shortage_cost_exposure']:.2f} shortage).\n"
        f"• Action Recommendation: Align kitchen production to {agent2_out['recommended_prep_quantity']} units to optimize order fulfillment while minimizing surplus food risk."
    )
    
    api_key = check_gemini_api_key()
    llm_explanation = None
    
    if api_key and HAS_GENAI:
        try:
            client = genai.Client(api_key=api_key)
            prompt = (
                f"You are the AI Operations Copilot for FoodWise AI (UN SDG 12 & SDG 2 aligned).\n"
                f"Synthesize the following deterministic operational metrics into a crisp, professional 3-bullet executive advice for the kitchen manager:\n"
                f"Data:\n"
                f"{json.dumps({'agent1': agent1_out, 'agent2': agent2_out, 'agent3': agent3_out}, indent=2)}\n"
                f"Be objective, explain uncertainty, and emphasize that cost exposures are scenario calculations based on user parameters."
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            llm_explanation = response.text
        except Exception as e:
            llm_explanation = f"(Gemini API call failed: {str(e)}. Displaying template fallback.)"

    return {
        "agent1_demand_analyst": agent1_out,
        "agent2_kitchen_planner": agent2_out,
        "agent3_risk_analyst": agent3_out,
        "agent4_operations_summary": summary_text,
        "llm_explanation": llm_explanation,
        "forecast_df": forecast_df
    }
