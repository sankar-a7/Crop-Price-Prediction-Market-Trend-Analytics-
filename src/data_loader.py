"""
Data Ingestion and Loading Module.
Loads historical commodity market records from disk or raw sources,
validates schema integrity, and sorts by temporal sequences.
"""

import os
from typing import Optional, List
import pandas as pd

REQUIRED_COLUMNS = [
    "date", "commodity", "category", "region", "market",
    "arrival_quantity_tonnes", "rainfall_mm", "fuel_price_index",
    "msp", "modal_price", "min_price", "max_price"
]

def load_raw_data(filepath: str = "data/raw/commodity_market_data.csv") -> pd.DataFrame:
    """
    Load raw commodity records and validate required schema columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Commodity dataset not found at: {filepath}. Run generate_dataset.py first.")
        
    df = pd.read_csv(filepath)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required schema columns: {missing_cols}")
        
    # Convert date and sort chronologically per market/commodity
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["commodity", "market", "date"]).reset_index(drop=True)
    return df

def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Generate structural summary of the ingested commodity market dataset.
    """
    return {
        "total_records": len(df),
        "date_range": (df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d")),
        "commodities": df["commodity"].unique().tolist(),
        "categories": df["category"].unique().tolist(),
        "regions": df["region"].unique().tolist(),
        "markets": df["market"].unique().tolist(),
        "missing_counts": df.isna().sum().to_dict(),
        "modal_price_range": (float(df["modal_price"].min()), float(df["modal_price"].max()))
    }
