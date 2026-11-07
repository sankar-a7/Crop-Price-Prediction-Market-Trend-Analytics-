"""
Domain-Specific Feature Engineering Module.
Extracts cyclical temporal signals, lagged autoregressive prices, rolling volatility,
supply-demand elasticity ratios, and agricultural season indicators.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd


class CommodityFeatureEngineer:
    """
    Transforms cleaned commodity market data into high-dimensional feature matrices
    engineered specifically for agricultural price forecasting.
    """

    def __init__(self):
        self.feature_columns: List[str] = []
        self.categorical_cols = ["commodity", "category", "region"]

    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive calendar, cyclical harmonics, and Indian crop calendar seasonality.
        """
        df_feat = df.copy()
        
        # Ensure datetime
        df_feat["date"] = pd.to_datetime(df_feat["date"])
        
        # Calendar units
        df_feat["year"] = df_feat["date"].dt.year
        df_feat["month"] = df_feat["date"].dt.month
        df_feat["week"] = df_feat["date"].dt.isocalendar().week.astype(int)
        df_feat["quarter"] = df_feat["date"].dt.quarter
        
        # Cyclical harmonics (Preserves circular continuity: Dec -> Jan)
        df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12.0)
        df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12.0)
        df_feat["week_sin"] = np.sin(2 * np.pi * df_feat["week"] / 52.0)
        df_feat["week_cos"] = np.cos(2 * np.pi * df_feat["week"] / 52.0)
        
        # Indian Agro-Climatic Seasons
        # Kharif: July (7) to October (10)
        df_feat["is_kharif"] = df_feat["month"].isin([7, 8, 9, 10]).astype(int)
        # Rabi: November (11) to April (4)
        df_feat["is_rabi"] = df_feat["month"].isin([11, 12, 1, 2, 3, 4]).astype(int)
        # Zaid: May (5) to June (6)
        df_feat["is_zaid"] = df_feat["month"].isin([5, 6]).astype(int)
        
        return df_feat

    def extract_lagged_and_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute autoregressive price lags and rolling statistical metrics within commodity-market series.
        """
        df_feat = df.copy()
        df_feat = df_feat.sort_values(by=["commodity", "market", "date"]).reset_index(drop=True)
        
        # Group object for commodity-market series
        grouped = df_feat.groupby(["commodity", "market"])
        
        # Autoregressive price lags (1 week, 2 weeks, 4 weeks, 8 weeks, 12 weeks, 52 weeks)
        lags = [1, 2, 4, 8, 12, 52]
        for lag in lags:
            df_feat[f"price_lag_{lag}w"] = grouped["modal_price"].shift(lag)
            
        # Arrival quantity lags
        df_feat["arrival_lag_1w"] = grouped["arrival_quantity_tonnes"].shift(1)
        df_feat["arrival_lag_4w"] = grouped["arrival_quantity_tonnes"].shift(4)
        
        # Rolling moving averages and volatility
        # 4-week window
        df_feat["price_rolling_mean_4w"] = grouped["modal_price"].transform(
            lambda s: s.shift(1).rolling(window=4, min_periods=1).mean()
        )
        df_feat["price_rolling_std_4w"] = grouped["modal_price"].transform(
            lambda s: s.shift(1).rolling(window=4, min_periods=1).std().fillna(0.0)
        )
        
        # 12-week window (Quarterly trend)
        df_feat["price_rolling_mean_12w"] = grouped["modal_price"].transform(
            lambda s: s.shift(1).rolling(window=12, min_periods=1).mean()
        )
        df_feat["price_rolling_std_12w"] = grouped["modal_price"].transform(
            lambda s: s.shift(1).rolling(window=12, min_periods=1).std().fillna(0.0)
        )
        
        # Backfill and forward fill lags first so derived ratios don't generate NaNs
        lag_cols = [c for c in df_feat.columns if "lag" in c or "rolling" in c]
        for col in lag_cols:
            df_feat[col] = grouped[col].transform(lambda s: s.bfill().ffill())
            df_feat[col] = df_feat[col].fillna(df_feat["modal_price"])
            
        # 4-week rolling arrival volume
        df_feat["arrival_rolling_mean_4w"] = grouped["arrival_quantity_tonnes"].transform(
            lambda s: s.shift(1).rolling(window=4, min_periods=1).mean()
        ).bfill().ffill()
        df_feat["arrival_rolling_mean_4w"] = df_feat["arrival_rolling_mean_4w"].fillna(df_feat["arrival_quantity_tonnes"])
        
        # Price Momentum: Ratio of 1-week prior price to 4-week moving average
        df_feat["price_momentum_4w"] = (
            df_feat["price_lag_1w"] / (df_feat["price_rolling_mean_4w"] + 1e-5)
        ).fillna(1.0)
        
        # Supply Elasticity Ratio: current arrivals vs recent rolling volume
        df_feat["supply_arrival_ratio"] = (
            df_feat["arrival_quantity_tonnes"] / (df_feat["arrival_rolling_mean_4w"] + 1e-5)
        ).fillna(1.0)
        
        # 4-week cumulative rainfall
        df_feat["rainfall_rolling_4w"] = grouped["rainfall_mm"].transform(
            lambda s: s.rolling(window=4, min_periods=1).mean()
        ).bfill().ffill().fillna(0.0)
        
        # Floor price buffer: Modal price relative to Government MSP
        df_feat["msp_buffer_ratio"] = np.where(
            df_feat["msp"] > 0,
            df_feat["price_lag_1w"] / (df_feat["msp"] + 1e-5),
            1.0
        )
        df_feat["msp_buffer_ratio"] = df_feat["msp_buffer_ratio"].fillna(1.0)
        
        return df_feat

    def encode_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        One-hot encode categorical features and return full feature matrix X and target y.
        """
        non_feature_cols = ["date", "modal_price", "min_price", "max_price", "market"]
        
        cat_cols = [c for c in self.categorical_cols if c in df.columns]
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)
        
        feature_cols = [c for c in df_encoded.columns if c not in non_feature_cols and not c.startswith("market")]
        self.feature_columns = feature_cols
        
        X = df_encoded[feature_cols].copy().astype(float)
        # Ensure zero remaining NaNs
        X = X.fillna(0.0)
        y = df["modal_price"].copy().astype(float)
        
        return df_encoded, X, y

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Full feature engineering pipeline execution.
        """
        df_temporal = self.extract_temporal_features(df)
        df_lags = self.extract_lagged_and_rolling_features(df_temporal)
        df_encoded, X, y = self.encode_features(df_lags)
        return df_lags, X, y
