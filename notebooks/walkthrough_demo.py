"""
Interactive End-to-End Walkthrough & Demonstration Script.
Demonstrates the full lifecycle from data ingestion to ML forecasting and explainability.
Can be executed directly or used during project presentations/viva.
"""

import os
import sys
# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import pandas as pd

from src.data_loader import load_raw_data, get_dataset_summary
from src.data_preprocessing import CommodityDataPreprocessor
from src.feature_engineering import CommodityFeatureEngineer
from src.models.random_forest import RandomForestPriceForecaster
from src.explainability import ModelExplainer
from predict import run_scenario_prediction


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"🚀 {title.upper()}")
    print("=" * 80)


def main():
    print_header("Step 1: Historical Commodity Market Data Ingestion")
    raw_df = load_raw_data("data/raw/commodity_market_data.csv")
    summary = get_dataset_summary(raw_df)
    print(f"Total Historical Records: {summary['total_records']}")
    print(f"Temporal Coverage:        {summary['date_range'][0]} to {summary['date_range'][1]}")
    print(f"Monitored Commodities:    {', '.join(summary['commodities'][:6])}... ({len(summary['commodities'])} crops)")
    print(f"Regional APMC Mandis:     {', '.join(summary['markets'][:4])}... ({len(summary['markets'])} mandis)")
    print(f"Raw Missing Values:       {summary['missing_counts']}")

    print_header("Step 2: Automated Imputation & Domain Outlier Removal")
    preprocessor = CommodityDataPreprocessor()
    clean_df, stats = preprocessor.fit_transform(raw_df)
    print(f"Imputed Non-Destructively: {stats['imputation']}")
    print(f"Outliers Filtered:        {stats['outliers']['outliers_removed']} records ({stats['outliers']['removal_percentage']}%)")
    print(f"Valid Analytical Dataset: {len(clean_df)} records")

    print_header("Step 3: Domain-Specific Feature Engineering")
    fe = CommodityFeatureEngineer()
    df_lags, X, y = fe.transform(clean_df)
    print(f"Engineered Feature Dimensions: {X.shape[1]} features across {X.shape[0]} samples")
    print(f"Feature Subsets: Cyclical Harmonics (month_sin, week_sin), Agro-Seasons (Kharif, Rabi, Zaid)")
    print(f"Autoregressive Lags: 1w, 2w, 4w, 8w, 12w, 52w (Year-over-Year)")
    print(f"Market Elasticity:   Arrival Ratios, MSP Buffers, Rolling Volatilities")

    print_header("Step 4: Machine Learning Benchmark Evaluation (2025 Test Set)")
    with open("models/benchmark_results.json", "r") as f:
        meta = json.load(f)
    print(f"{'MODEL ARCHITECTURE':<30} | {'RMSE (INR)':<12} | {'MAE (INR)':<10} | {'R² SCORE':<10} | {'TREND ACC (%)':<12}")
    print("-" * 80)
    for model_name, m in meta["overall_benchmarks"].items():
        print(f"{model_name:<30} | {m['rmse']:<12.2f} | {m['mae']:<10.2f} | {m['r2']:<10.4f} | {m['directional_trend_accuracy']:<12.2f}%")
    print("-" * 80)
    print(f"⭐ Regional Category Average Trend Accuracy: {meta['regional_average_trend_accuracy']}%\n")

    print_header("Step 5: 'What-If' Supply Shock Scenario Simulation & Explainability")
    print("Scenario: Simulating a 30% harvest shortfall in Lasalgaon Onion Mandi due to unseasonal rainfall...")
    sim_res = run_scenario_prediction(
        commodity="Onion",
        market="Lasalgaon (Maharashtra)",
        region="West",
        category="Vegetables",
        arrival_quantity=4200.0,
        rainfall_mm=65.0,
        fuel_index=150.0,
        msp=0.0,
        recent_price=1850.0,
        model_choice="Random Forest"
    )
    print(f"Commodity:            {sim_res['commodity']} ({sim_res['market']})")
    print(f"Baseline Price:       INR {sim_res['input_recent_price']:.2f} / quintal")
    print(f"Forecasted Price:     INR {sim_res['forecasted_modal_price']:.2f} / quintal")
    print(f"Projected Shift:      INR {sim_res['price_change']:+.2f} ({sim_res['percentage_change']:+.2f}%)")
    print(f"Anticipated Trend:    {sim_res['anticipated_trend']}")

    # Model Explainability
    rf_model = RandomForestPriceForecaster.load("models/random_forest.joblib")
    explainer = ModelExplainer(rf_model, meta["feature_names"])
    explanation = explainer.explain_instance_delta(
        baseline_price=1850.0,
        predicted_price=sim_res["forecasted_modal_price"],
        feature_dict={"supply_arrival_ratio": 0.70, "fuel_price_index": 150.0, "msp": 0.0}
    )
    print("\nFeature Attribution Breakdown:")
    for attr in explanation["domain_attributions"]:
        print(f"  - {attr['domain']:<35}: {attr['impact_inr']:+.2f} INR ({attr['direction']})")

    print_header("Walkthrough Completed Successfully!")


if __name__ == "__main__":
    main()
