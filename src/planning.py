import pandas as pd
import numpy as np

def simulate_surplus_and_shortage(
    forecast_demand,
    planned_quantity,
    service_buffer_pct=10.0,
    cost_per_unsold=50.0,
    cost_per_shortage=120.0,
    currency_symbol="₹"
):
    """
    Simulates operational surplus and shortage risks under planned prep vs ML forecast.
    Calculates scenario cost exposures based on user inputs.
    
    Formulas:
      - Recommended Prep = round(Forecast Demand * (1 + Service Buffer % / 100))
      - Potential Surplus = max(Planned Prep - Forecast Demand, 0)
      - Potential Shortage = max(Forecast Demand - Planned Prep, 0)
      - Surplus Exposure = Potential Surplus * Cost per Unsold Unit
      - Shortage Exposure = Potential Shortage * Penalty per Unfulfilled Unit
      - Total Scenario Exposure = Surplus Exposure + Shortage Exposure
    """
    forecast_demand = float(forecast_demand)
    planned_quantity = float(planned_quantity)
    service_buffer_pct = float(service_buffer_pct)
    cost_per_unsold = float(cost_per_unsold)
    cost_per_shortage = float(cost_per_shortage)

    # Transparent recommended prep quantity
    recommended_quantity = round(forecast_demand * (1.0 + service_buffer_pct / 100.0))
    
    # Potential surplus & shortage
    potential_surplus = max(planned_quantity - forecast_demand, 0.0)
    potential_shortage = max(forecast_demand - planned_quantity, 0.0)
    
    # Scenario cost exposure
    surplus_cost = potential_surplus * cost_per_unsold
    shortage_cost = potential_shortage * cost_per_shortage
    total_cost_exposure = surplus_cost + shortage_cost

    return {
        "forecast_demand": round(forecast_demand),
        "planned_quantity": round(planned_quantity),
        "recommended_quantity": recommended_quantity,
        "service_buffer_pct": service_buffer_pct,
        "potential_surplus_units": round(potential_surplus),
        "potential_shortage_units": round(potential_shortage),
        "surplus_cost_exposure": round(surplus_cost, 2),
        "shortage_cost_exposure": round(shortage_cost, 2),
        "total_cost_exposure": round(total_cost_exposure, 2),
        "cost_per_unsold": cost_per_unsold,
        "cost_per_shortage": cost_per_shortage,
        "currency_symbol": currency_symbol
    }

def generate_plan_comparison(forecast_demand, current_plan, service_buffer_pct, cost_per_unsold, cost_per_shortage, currency_symbol="₹"):
    """
    Generates a comparison grid across multiple operational strategy scenarios.
    """
    scenarios = {
        "Current Plan": current_plan,
        "ML Recommended (Forecast + Buffer)": round(forecast_demand * (1.0 + service_buffer_pct / 100.0)),
        "Exact ML Forecast": round(forecast_demand),
        "Conservative Plan (Forecast - 10%)": round(forecast_demand * 0.90),
        "Aggressive Plan (Forecast + 20%)": round(forecast_demand * 1.20),
    }

    comparison = []
    for plan_name, quantity in scenarios.items():
        sim = simulate_surplus_and_shortage(
            forecast_demand=forecast_demand,
            planned_quantity=quantity,
            service_buffer_pct=service_buffer_pct,
            cost_per_unsold=cost_per_unsold,
            cost_per_shortage=cost_per_shortage,
            currency_symbol=currency_symbol
        )
        comparison.append({
            "Plan Scenario": plan_name,
            "Prep Quantity (Units)": sim["planned_quantity"],
            "Potential Surplus (Units)": sim["potential_surplus_units"],
            "Potential Shortage (Units)": sim["potential_shortage_units"],
            f"Surplus Cost ({currency_symbol})": sim["surplus_cost_exposure"],
            f"Shortage Cost ({currency_symbol})": sim["shortage_cost_exposure"],
            f"Total Scenario Exposure ({currency_symbol})": sim["total_cost_exposure"]
        })

    return pd.DataFrame(comparison)
