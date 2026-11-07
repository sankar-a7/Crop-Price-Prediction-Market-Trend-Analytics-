"""
Training & Benchmarking CLI Entrypoint.
Executes the full pipeline, benchmarks Random Forest, XGBoost / Gradient Boosting, and DNN,
and serializes models and metrics.
"""

import argparse
import sys
from src.pipeline import ForecastingPipeline


def main():
    parser = argparse.ArgumentParser(description="Crop Price Prediction & Market Trend Forecasting Training Pipeline")
    parser.add_argument("--raw-data", type=str, default="data/raw/commodity_market_data.csv", help="Path to raw dataset")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning with TimeSeriesSplit")
    args = parser.parse_args()

    print("================================================================================")
    print("Crop Price Prediction & Market Trend Analytics — End-to-End ML Pipeline")
    print("================================================================================\n")

    pipeline = ForecastingPipeline(raw_data_path=args.raw_data)
    results = pipeline.train_and_benchmark(tune_hyperparameters=args.tune)

    print("\n" + "=" * 80)
    print(f"{'MODEL ARCHITECTURE':<30} | {'RMSE':<10} | {'MAE':<10} | {'R² SCORE':<10} | {'TREND ACC (%)':<12}")
    print("-" * 80)
    for model_name, metrics in results["overall_benchmarks"].items():
        print(f"{model_name:<30} | {metrics['rmse']:<10.2f} | {metrics['mae']:<10.2f} | {metrics['r2']:<10.4f} | {metrics.get('directional_trend_accuracy', 0):<12.2f}%")
    print("=" * 80)
    print(f"Regional Average Directional Trend Accuracy: {results['regional_average_trend_accuracy']}%\n")

if __name__ == "__main__":
    main()

