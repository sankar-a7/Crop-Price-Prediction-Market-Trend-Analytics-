"""
Data Preprocessing Pipeline Module.
Implements automated missing-value imputation and domain-aware outlier removal.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd


class CommodityDataPreprocessor:
    """
    Automated preprocessing pipeline for agricultural commodity market data.
    Handles temporal missing-value imputation and domain-specific outlier filtering.
    """

    def __init__(self, iqr_multiplier: float = 2.5):
        self.iqr_multiplier = iqr_multiplier
        self.stats_: Dict[str, Any] = {}

    def impute_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Impute missing values using multi-stage hierarchical strategy:
        1. Temporal linear interpolation within each commodity-market time series.
        2. Group forward/backward fill.
        3. Commodity-regional median imputation.
        4. Global fallback.
        """
        df_imputed = df.copy()
        imputation_report = {}
        
        target_cols = ["arrival_quantity_tonnes", "rainfall_mm", "fuel_price_index", "modal_price"]
        
        for col in target_cols:
            if col not in df_imputed.columns:
                continue
            init_nans = int(df_imputed[col].isna().sum())
            if init_nans > 0:
                # Stage 1: Time-series interpolation per market series
                df_imputed[col] = df_imputed.groupby(["commodity", "market"])[col].transform(
                    lambda g: g.interpolate(method="linear", limit_direction="both")
                )
                
                # Stage 2: Forward & backward fill
                df_imputed[col] = df_imputed.groupby(["commodity", "region"])[col].transform(
                    lambda g: g.ffill().bfill()
                )
                
                # Stage 3: Commodity median fallback
                df_imputed[col] = df_imputed.groupby("commodity")[col].transform(
                    lambda g: g.fillna(g.median())
                )
                
                # Stage 4: Global median
                if df_imputed[col].isna().sum() > 0:
                    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
                    
            imputation_report[col] = init_nans

        return df_imputed, imputation_report

    def remove_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Detect and filter recording outliers using agricultural domain thresholds and IQR bounds.
        Preserves natural seasonal price surges while eliminating data entry errors.
        """
        df_clean = df.copy()
        initial_count = len(df_clean)
        
        # 1. Eliminate non-positive prices and impossible quantities
        valid_mask = (df_clean["modal_price"] > 50.0) & (df_clean["arrival_quantity_tonnes"] > 0)
        df_clean = df_clean[valid_mask].copy()
        
        # 2. Group-specific IQR bounding on modal price
        outlier_indices = []
        for commodity, group in df_clean.groupby("commodity"):
            q1 = group["modal_price"].quantile(0.25)
            q3 = group["modal_price"].quantile(0.75)
            iqr = q3 - q1
            lower_bound = max(50.0, q1 - self.iqr_multiplier * iqr)
            upper_bound = q3 + self.iqr_multiplier * iqr
            
            # Identify indices outside reasonable bounds
            anomalies = group[(group["modal_price"] < lower_bound) | (group["modal_price"] > upper_bound)].index
            outlier_indices.extend(anomalies)

        # 3. Filter extreme arrival outliers (e.g. sensor 999999 errors)
        for commodity, group in df_clean.groupby("commodity"):
            q3_arr = group["arrival_quantity_tonnes"].quantile(0.75)
            iqr_arr = q3_arr - group["arrival_quantity_tonnes"].quantile(0.25)
            arr_upper = q3_arr + 4.0 * iqr_arr
            arr_anomalies = group[group["arrival_quantity_tonnes"] > arr_upper].index
            outlier_indices.extend(arr_anomalies)

        outlier_indices = list(set(outlier_indices))
        df_clean = df_clean.drop(index=outlier_indices).reset_index(drop=True)
        
        removed_count = initial_count - len(df_clean)
        outlier_report = {
            "initial_records": initial_count,
            "outliers_removed": removed_count,
            "cleaned_records": len(df_clean),
            "removal_percentage": round((removed_count / initial_count) * 100, 2)
        }
        
        return df_clean, outlier_report

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run the full data preprocessing workflow and return cleaned data and audit report.
        """
        imputed_df, imp_report = self.impute_missing_values(df)
        cleaned_df, out_report = self.remove_outliers(imputed_df)
        
        self.stats_ = {
            "imputation": imp_report,
            "outliers": out_report
        }
        return cleaned_df, self.stats_

