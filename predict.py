"""
Inference and What-If Scenario Prediction CLI.
Loads trained models and runs single or batch forecasting scenarios with supply-chain shock simulations.
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import joblib
from src.models.random_forest import RandomForestPriceForecaster
from src.models.gradient_boost import GradientBoostedPriceForecaster
from src.models.neural_network import DeepNeuralNetworkPriceForecaster


def run_scenario_prediction(
    commodity: str,
    market: str,
    region: str,
    category: str,
    arrival_quantity: float,
    rainfall_mm: float,
    fuel_index: float,
    msp: float,
    recent_price: float,
    model_choice: str = "Random Forest"
) -> dict:
    """
    Run one-shot forecasting with customizable shock parameters.
    """
    # Load feature metadata
    with open("models/benchmark_results.json", "r") as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    
    # Construct base feature vector matching training schema
    # Use standard default or derived seasonal harmonics (assume current month)
    import datetime
    today = datetime.date.today()
    month = today.month
    week = today.isocalendar()[1]
    
    feature_dict = {feat: 0.0 for feat in feature_names}
    
    # Set numeric drivers
    feature_dict["arrival_quantity_tonnes"] = float(arrival_quantity)
    feature_dict["rainfall_mm"] = float(rainfall_mm)
    feature_dict["fuel_price_index"] = float(fuel_index)
    feature_dict["msp"] = float(msp)
    
    # Calendar & cyclical
    feature_dict["year"] = today.year
    feature_dict["month"] = month
    feature_dict["week"] = week
    feature_dict["quarter"] = (month - 1) // 3 + 1
    feature_dict["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    feature_dict["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    feature_dict["week_sin"] = np.sin(2 * np.pi * week / 52.0)
    feature_dict["week_cos"] = np.cos(2 * np.pi * week / 52.0)
    feature_dict["is_kharif"] = 1.0 if month in [7, 8, 9, 10] else 0.0
    feature_dict["is_rabi"] = 1.0 if month in [11, 12, 1, 2, 3, 4] else 0.0
    feature_dict["is_zaid"] = 1.0 if month in [5, 6] else 0.0
    
    # Lags & rolling
    feature_dict["price_lag_1w"] = float(recent_price)
    feature_dict["price_lag_2w"] = float(recent_price) * 0.99
    feature_dict["price_lag_4w"] = float(recent_price) * 0.98
    feature_dict["price_lag_8w"] = float(recent_price) * 0.96
    feature_dict["price_lag_12w"] = float(recent_price) * 0.95
    feature_dict["price_lag_52w"] = float(recent_price) * 0.92
    
    feature_dict["arrival_lag_1w"] = float(arrival_quantity)
    feature_dict["arrival_lag_4w"] = float(arrival_quantity)
    feature_dict["price_rolling_mean_4w"] = float(recent_price)
    feature_dict["price_rolling_std_4w"] = float(recent_price) * 0.04
    feature_dict["price_rolling_mean_12w"] = float(recent_price) * 0.98
    feature_dict["price_rolling_std_12w"] = float(recent_price) * 0.05
    feature_dict["price_momentum_4w"] = 1.0
    feature_dict["arrival_rolling_mean_4w"] = float(arrival_quantity)
    feature_dict["supply_arrival_ratio"] = 1.0
    feature_dict["rainfall_rolling_4w"] = float(rainfall_mm)
    feature_dict["msp_buffer_ratio"] = float(recent_price) / (msp + 1e-5) if msp > 0 else 1.0
    
    # Set one-hot flags
    for key in [f"commodity_{commodity}", f"category_{category}", f"region_{region}"]:
        if key in feature_dict:
            feature_dict[key] = 1.0
            
    df_row = pd.DataFrame([feature_dict])
    
    # Load selected model
    if model_choice == "Random Forest":
        model = RandomForestPriceForecaster.load("models/random_forest.joblib")
        pred_price = float(model.predict(df_row)[0])
    elif model_choice == "XGBoost / Gradient Boosting":
        model = GradientBoostedPriceForecaster.load("models/xgboost_regressor.joblib")
        pred_price = float(model.predict(df_row)[0])
    else:
        model = DeepNeuralNetworkPriceForecaster.load("models/deep_neural_network.pt")
        pred_price = float(model.predict(df_row)[0])
        
    delta = pred_price - recent_price
    pct_change = (delta / recent_price) * 100
    trend = "BULLISH (UP)" if pct_change > 1.5 else ("BEARISH (DOWN)" if pct_change < -1.5 else "NEUTRAL (SIDEWAYS)")
    
    return {
        "commodity": commodity,
        "market": market,
        "region": region,
        "model_used": model_choice,
        "input_recent_price": recent_price,
        "forecasted_modal_price": round(pred_price, 2),
        "price_change": round(delta, 2),
        "percentage_change": round(pct_change, 2),
        "anticipated_trend": trend,
        "msp_floor": msp,
        "above_msp": pred_price >= msp if msp > 0 else None
    }


def main():
    parser = argparse.ArgumentParser(description="Commodity Price Forecasting CLI")
    parser.add_argument("--commodity", type=str, default="Wheat", help="Commodity name")
    parser.add_argument("--region", type=str, default="North", help="Region")
    parser.add_argument("--category", type=str, default="Cereals", help="Category")
    parser.add_argument("--market", type=str, default="Khanna (Punjab)", help="Market/Mandi")
    parser.add_argument("--recent-price", type=float, default=2400.0, help="Current market price (INR/quintal)")
    parser.add_argument("--arrivals", type=float, default=4500.0, help="Arrival quantity in tonnes")
    parser.add_argument("--rainfall", type=float, default=25.0, help="Recent weekly rainfall (mm)")
    parser.add_argument("--fuel-index", type=float, default=145.0, help="Fuel price index (base=100)")
    parser.add_argument("--msp", type=float, default=2275.0, help="Government MSP (INR/quintal)")
    parser.add_argument("--model", type=str, default="Random Forest", choices=["Random Forest", "XGBoost / Gradient Boosting", "Deep Neural Network"])
    
    args = parser.parse_args()
    
    result = run_scenario_prediction(
        commodity=args.commodity,
        market=args.market,
        region=args.region,
        category=args.category,
        arrival_quantity=args.arrivals,
        rainfall_mm=args.rainfall,
        fuel_index=args.fuel_index,
        msp=args.msp,
        recent_price=args.recent_price,
        model_choice=args.model
    )
    
    print("\n" + "=" * 60)
    print("🌾 CROP PRICE PREDICTION & TREND FORECAST REPORT")
    print("=" * 60)
    print(f"Commodity:           {result['commodity']} ({result['market']} - {result['region']})")
    print(f"Model Architecture:  {result['model_used']}")
    print(f"Baseline Price:      ₹{result['input_recent_price']:.2f} / quintal")
    print(f"Forecasted Price:    ₹{result['forecasted_modal_price']:.2f} / quintal")
    print(f"Expected Shift:      ₹{result['price_change']:+.2f} ({result['percentage_change']:+.2f}%)")
    print(f"Anticipated Trend:   {result['anticipated_trend']}")
    if result['msp_floor'] > 0:
        print(f"MSP Safety Margin:   {'Safe (+₹' + str(round(result['forecasted_modal_price'] - result['msp_floor'], 2)) + ')' if result['above_msp'] else 'Below MSP Floor'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

