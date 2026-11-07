"""
Unit and Integration Test Suite for Crop Price Prediction Pipeline.
Validates ingestion, imputation, outlier cleaning, feature engineering, and inference integrity.
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.data_loader import load_raw_data, get_dataset_summary
from src.data_preprocessing import CommodityDataPreprocessor
from src.feature_engineering import CommodityFeatureEngineer
from src.evaluation import calculate_metrics
from src.models.random_forest import RandomForestPriceForecaster
from src.models.gradient_boost import GradientBoostedPriceForecaster
from src.models.neural_network import DeepNeuralNetworkPriceForecaster


class TestCropPriceForecastingPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raw_data_path = "data/raw/commodity_market_data.csv"
        assert os.path.exists(cls.raw_data_path), "Raw dataset must exist for tests."
        cls.df = load_raw_data(cls.raw_data_path)

    def test_01_data_loading_and_schema(self):
        """Test dataset loading, row counts, and required columns."""
        self.assertGreater(len(self.df), 1000)
        required_cols = [
            "date", "commodity", "category", "region", "market",
            "arrival_quantity_tonnes", "rainfall_mm", "fuel_price_index",
            "msp", "modal_price"
        ]
        for col in required_cols:
            self.assertIn(col, self.df.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.df["date"]))

    def test_02_missing_value_imputation(self):
        """Verify automated multi-stage imputation handles nulls completely."""
        preprocessor = CommodityDataPreprocessor()
        df_imputed, report = preprocessor.impute_missing_values(self.df)
        
        self.assertEqual(df_imputed["arrival_quantity_tonnes"].isna().sum(), 0)
        self.assertEqual(df_imputed["rainfall_mm"].isna().sum(), 0)
        self.assertEqual(df_imputed["modal_price"].isna().sum(), 0)

    def test_03_outlier_removal(self):
        """Ensure impossible prices and extreme typos are filtered."""
        preprocessor = CommodityDataPreprocessor()
        df_clean, report = preprocessor.remove_outliers(self.df)
        
        # All remaining modal prices must be positive and within reasonable bounds
        self.assertTrue((df_clean["modal_price"] > 50.0).all())
        self.assertGreater(report["outliers_removed"], 0)
        self.assertLess(report["removal_percentage"], 5.0)

    def test_04_feature_engineering_integrity(self):
        """Validate cyclical encoding, autoregressive lags, and NaN-free feature matrix."""
        preprocessor = CommodityDataPreprocessor()
        cleaned_df, _ = preprocessor.fit_transform(self.df)
        
        fe = CommodityFeatureEngineer()
        df_lags, X, y = fe.transform(cleaned_df)
        
        # Cyclical bounds: sin/cos must strictly stay in [-1, 1]
        self.assertTrue((X["month_sin"] >= -1.0).all() and (X["month_sin"] <= 1.0).all())
        self.assertTrue((X["month_cos"] >= -1.0).all() and (X["month_cos"] <= 1.0).all())
        
        # Zero residual NaNs
        self.assertEqual(X.isna().sum().sum(), 0, "Feature matrix X must contain 0 NaNs.")
        self.assertEqual(len(X), len(y))

    def test_05_evaluation_metrics_and_trend_accuracy(self):
        """Verify metric calculation including directional trend accuracy."""
        y_true = np.array([2000.0, 2100.0, 2050.0, 2200.0])
        y_pred = np.array([2010.0, 2090.0, 2040.0, 2210.0])
        y_lag = np.array([1950.0, 2000.0, 2100.0, 2050.0])
        
        metrics = calculate_metrics(y_true, y_pred, y_lag)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)
        self.assertIn("directional_trend_accuracy", metrics)
        self.assertGreater(metrics["r2"], 0.90)
        self.assertEqual(metrics["directional_trend_accuracy"], 100.0)

    def test_06_model_loading_and_inference(self):
        """Test serialized model loading and forward prediction sanity."""
        rf = RandomForestPriceForecaster.load("models/random_forest.joblib")
        gb = GradientBoostedPriceForecaster.load("models/xgboost_regressor.joblib")
        dnn = DeepNeuralNetworkPriceForecaster.load("models/deep_neural_network.pt")
        
        # Use full engineered feature set sample
        fe = CommodityFeatureEngineer()
        cleaned_df, _ = CommodityDataPreprocessor().fit_transform(self.df)
        _, X, _ = fe.transform(cleaned_df)
        sample_x = X.head(20)
        
        for m in [rf, gb, dnn]:
            aligned_x = sample_x.reindex(columns=m.feature_names_, fill_value=0.0)
            preds = m.predict(aligned_x)
            self.assertEqual(len(preds), len(sample_x))
            self.assertTrue(np.all(preds > 0), "Predicted prices must all be positive.")


if __name__ == "__main__":
    unittest.main()
