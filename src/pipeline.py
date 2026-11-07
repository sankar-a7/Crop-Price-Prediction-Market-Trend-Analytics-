"""
Unified End-to-End ML Forecasting Pipeline.
Orchestrates ingestion, preprocessing, feature engineering, temporal validation,
model training, hyperparameter optimization, benchmarking, and artifact serialization.
"""

import os
import json
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.data_loader import load_raw_data, get_dataset_summary
from src.data_preprocessing import CommodityDataPreprocessor
from src.feature_engineering import CommodityFeatureEngineer
from src.models.random_forest import RandomForestPriceForecaster
from src.models.gradient_boost import GradientBoostedPriceForecaster
from src.models.neural_network import DeepNeuralNetworkPriceForecaster
from src.evaluation import calculate_metrics, benchmark_regional_categories


class ForecastingPipeline:
    """
    Complete end-to-end forecasting pipeline management.
    """

    def __init__(
        self,
        raw_data_path: str = "data/raw/commodity_market_data.csv",
        processed_data_path: str = "data/processed/processed_features.csv",
        models_dir: str = "models"
    ):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        
        self.preprocessor = CommodityDataPreprocessor()
        self.feature_engineer = CommodityFeatureEngineer()
        
        self.models = {
            "Random Forest": RandomForestPriceForecaster(n_estimators=160, max_depth=16),
            "XGBoost / Gradient Boosting": GradientBoostedPriceForecaster(learning_rate=0.06, n_estimators=160, max_depth=6),
            "Deep Neural Network": DeepNeuralNetworkPriceForecaster(epochs=55, batch_size=64, learning_rate=0.005)
        }
        self.benchmarks_: Dict[str, Any] = {}

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Load raw records, impute missing values, filter outliers, and engineer features.
        """
        print("[1/5] Ingesting raw market records...")
        raw_df = load_raw_data(self.raw_data_path)
        
        print("[2/5] Running automated imputation & domain outlier filtering...")
        cleaned_df, prep_stats = self.preprocessor.fit_transform(raw_df)
        print(f"      - Imputed features: {prep_stats['imputation']}")
        print(f"      - Outliers removed: {prep_stats['outliers']['outliers_removed']} records ({prep_stats['outliers']['removal_percentage']}%)")
        
        print("[3/5] Engineering cyclical, lag, rolling and market elasticity features...")
        df_lags, X, y = self.feature_engineer.transform(cleaned_df)
        
        # Save processed features for fast dashboard and scenario analysis
        df_lags.to_csv(self.processed_data_path, index=False)
        print(f"      - Processed features saved to: {self.processed_data_path}")
        
        return df_lags, X, y

    def train_and_benchmark(self, tune_hyperparameters: bool = False) -> Dict[str, Any]:
        """
        Train all regression architectures, benchmark metrics, and save serialized models.
        """
        df_lags, X, y = self.prepare_data()
        
        # Chronological Train-Test Split (2020-2024 for Training; 2025 as Out-of-Time Test Set)
        train_mask = df_lags["year"] < 2025
        test_mask = df_lags["year"] >= 2025
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        test_meta = df_lags[test_mask].copy()
        
        print(f"[4/5] Temporal Partitioning:")
        print(f"      - Training Set (2020-2024): {len(X_train)} samples")
        print(f"      - Holdout Test Set (2025): {len(X_test)} samples")
        
        benchmarks = {}
        predictions_df = pd.DataFrame({
            "date": test_meta["date"],
            "commodity": test_meta["commodity"],
            "category": test_meta["category"],
            "region": test_meta["region"],
            "market": test_meta["market"],
            "actual": y_test.values,
            "price_lag_1w": test_meta["price_lag_1w"].values
        })
        
        print("[5/5] Training & Benchmarking Regression Architectures...")
        
        for name, model in self.models.items():
            print(f"      -> Training {name}...")
            if tune_hyperparameters and hasattr(model, "tune_hyperparameters"):
                print(f"         Tuning hyperparameters for {name}...")
                model.tune_hyperparameters(X_train, y_train)
            else:
                model.fit(X_train, y_train)
                
            y_pred = model.predict(X_test)
            predictions_df[f"pred_{name}"] = y_pred
            
            # Calculate performance metrics
            metrics = calculate_metrics(
                y_true=y_test.values,
                y_pred=y_pred,
                y_lag=test_meta["price_lag_1w"].values
            )
            benchmarks[name] = metrics
            print(f"         Results -> RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f} | R2: {metrics['r2']:.4f} | Trend Acc: {metrics.get('directional_trend_accuracy', 0):.2f}%")
            
        # Regional breakdown for top performing model (Random Forest / Gradient Boost)
        best_model_name = min(benchmarks.keys(), key=lambda k: benchmarks[k]["rmse"])
        regional_df = benchmark_regional_categories(
            predictions_df,
            actual_col="actual",
            pred_col=f"pred_{best_model_name}",
            lag_col="price_lag_1w"
        )
        
        # Save models to disk
        rf_path = os.path.join(self.models_dir, "random_forest.joblib")
        gb_path = os.path.join(self.models_dir, "xgboost_regressor.joblib")
        dnn_path = os.path.join(self.models_dir, "deep_neural_network.pt")
        
        self.models["Random Forest"].save(rf_path)
        self.models["XGBoost / Gradient Boosting"].save(gb_path)
        self.models["Deep Neural Network"].save(dnn_path)
        
        # Save predictions and benchmarks
        pred_path = os.path.join(self.models_dir, "test_predictions.csv")
        predictions_df.to_csv(pred_path, index=False)
        
        regional_path = os.path.join(self.models_dir, "regional_benchmarks.csv")
        regional_df.to_csv(regional_path, index=False)
        
        metadata = {
            "best_model": best_model_name,
            "overall_benchmarks": benchmarks,
            "feature_names": list(X.columns),
            "regional_average_trend_accuracy": round(float(regional_df["directional_accuracy"].mean()), 2)
        }
        
        meta_path = os.path.join(self.models_dir, "benchmark_results.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        self.benchmarks_ = metadata
        print("\nPipeline execution complete! All artifacts saved successfully.")
        return metadata
