"""
Realistic Agricultural Commodity Historical Dataset Generator (2020 - 2025).
Generates high-fidelity time-series commodity market records across Indian regional mandis
incorporating seasonality, supply-demand arrival dynamics, weather patterns, MSP floors,
transport cost indices, and authentic market anomalies.
"""

import os
import numpy as np
import pandas as pd

def generate_market_data(output_path: str = "data/raw/commodity_market_data.csv", seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    # Date range: weekly records from Jan 2020 to Dec 2025 (~312 weeks)
    dates = pd.date_range(start="2020-01-05", end="2025-12-28", freq="W-SUN")
    n_weeks = len(dates)
    
    # Commodity specifications with baseline pricing, seasonality, and regional markets
    commodity_configs = [
        {
            "commodity": "Wheat",
            "category": "Cereals",
            "region": "North",
            "market": "Khanna (Punjab)",
            "base_price": 1925.0,
            "msp_base": 1925.0,
            "msp_growth": 0.05,
            "season_peak_month": 4,      # April harvest peak (low price)
            "lean_month": 12,            # Dec lean supply (high price)
            "seasonal_amplitude": 0.16,
            "base_arrivals": 4500,
            "arrival_season_mult": 3.2,
        },
        {
            "commodity": "Wheat",
            "category": "Cereals",
            "region": "Central",
            "market": "Indore (MP)",
            "base_price": 1975.0,
            "msp_base": 1925.0,
            "msp_growth": 0.05,
            "season_peak_month": 3,      # March harvest
            "lean_month": 11,
            "seasonal_amplitude": 0.17,
            "base_arrivals": 4200,
            "arrival_season_mult": 3.0,
        },
        {
            "commodity": "Paddy (Rice)",
            "category": "Cereals",
            "region": "South",
            "market": "Nizamabad (Telangana)",
            "base_price": 1868.0,
            "msp_base": 1868.0,
            "msp_growth": 0.052,
            "season_peak_month": 11,     # Nov Kharif harvest
            "lean_month": 6,             # June pre-harvest
            "seasonal_amplitude": 0.15,
            "base_arrivals": 5200,
            "arrival_season_mult": 2.7,
        },
        {
            "commodity": "Paddy (Rice)",
            "category": "Cereals",
            "region": "East",
            "market": "Burdwan (West Bengal)",
            "base_price": 1890.0,
            "msp_base": 1868.0,
            "msp_growth": 0.052,
            "season_peak_month": 12,
            "lean_month": 7,
            "seasonal_amplitude": 0.15,
            "base_arrivals": 4800,
            "arrival_season_mult": 2.6,
        },
        {
            "commodity": "Maize",
            "category": "Cereals",
            "region": "South",
            "market": "Davanagere (Karnataka)",
            "base_price": 1850.0,
            "msp_base": 1850.0,
            "msp_growth": 0.048,
            "season_peak_month": 10,
            "lean_month": 4,
            "seasonal_amplitude": 0.18,
            "base_arrivals": 3100,
            "arrival_season_mult": 2.5,
        },
        {
            "commodity": "Chickpea (Chana)",
            "category": "Pulses",
            "region": "West",
            "market": "Latur (Maharashtra)",
            "base_price": 4875.0,
            "msp_base": 4875.0,
            "msp_growth": 0.055,
            "season_peak_month": 3,      # March harvest
            "lean_month": 10,            # Festive demand
            "seasonal_amplitude": 0.20,
            "base_arrivals": 2500,
            "arrival_season_mult": 2.6,
        },
        {
            "commodity": "Pigeon Pea (Tur)",
            "category": "Pulses",
            "region": "South",
            "market": "Gulbarga (Karnataka)",
            "base_price": 5800.0,
            "msp_base": 5800.0,
            "msp_growth": 0.06,
            "season_peak_month": 1,      # Jan harvest
            "lean_month": 9,             # Festive peak
            "seasonal_amplitude": 0.22,
            "base_arrivals": 2200,
            "arrival_season_mult": 2.4,
        },
        {
            "commodity": "Soybean",
            "category": "Oilseeds",
            "region": "Central",
            "market": "Ujjain (MP)",
            "base_price": 3710.0,
            "msp_base": 3710.0,
            "msp_growth": 0.058,
            "season_peak_month": 10,     # Oct harvest
            "lean_month": 5,             # May lean
            "seasonal_amplitude": 0.22,
            "base_arrivals": 3800,
            "arrival_season_mult": 3.0,
        },
        {
            "commodity": "Mustard",
            "category": "Oilseeds",
            "region": "North",
            "market": "Jaipur (Rajasthan)",
            "base_price": 4425.0,
            "msp_base": 4425.0,
            "msp_growth": 0.062,
            "season_peak_month": 3,      # March harvest
            "lean_month": 11,            # Winter oil demand
            "seasonal_amplitude": 0.20,
            "base_arrivals": 3200,
            "arrival_season_mult": 2.8,
        },
        {
            "commodity": "Cotton",
            "category": "Cash Crops",
            "region": "West",
            "market": "Rajkot (Gujarat)",
            "base_price": 5515.0,
            "msp_base": 5515.0,
            "msp_growth": 0.055,
            "season_peak_month": 11,     # Nov picking
            "lean_month": 6,             # Lean summer
            "seasonal_amplitude": 0.19,
            "base_arrivals": 2800,
            "arrival_season_mult": 2.8,
        },
        {
            "commodity": "Onion",
            "category": "Vegetables",
            "region": "West",
            "market": "Lasalgaon (Maharashtra)",
            "base_price": 1600.0,
            "msp_base": 0.0,             # No MSP for onion
            "msp_growth": 0.0,
            "season_peak_month": 3,      # Rabi harvest supply glut
            "lean_month": 9,             # Sep-Nov lean supply spike
            "seasonal_amplitude": 0.42,    # High perishable seasonal swing
            "base_arrivals": 6000,
            "arrival_season_mult": 2.6,
        },
        {
            "commodity": "Potato",
            "category": "Vegetables",
            "region": "North",
            "market": "Agra (UP)",
            "base_price": 1100.0,
            "msp_base": 0.0,
            "msp_growth": 0.0,
            "season_peak_month": 2,      # Feb harvest
            "lean_month": 10,            # Pre-Diwali cold store release
            "seasonal_amplitude": 0.35,
            "base_arrivals": 7500,
            "arrival_season_mult": 3.0,
        },
        {
            "commodity": "Tomato",
            "category": "Vegetables",
            "region": "South",
            "market": "Kolar (Karnataka)",
            "base_price": 1400.0,
            "msp_base": 0.0,
            "msp_growth": 0.0,
            "season_peak_month": 1,      # Winter harvest
            "lean_month": 7,             # July monsoon flush / shortage
            "seasonal_amplitude": 0.45,    # High perishable swing
            "base_arrivals": 4500,
            "arrival_season_mult": 2.4,
        }
    ]
    
    # Fuel index: starts at 100 in Jan 2020 and trends upward to ~148 by late 2025
    fuel_trend = np.linspace(100.0, 146.0, n_weeks) + np.random.normal(0, 0.8, n_weeks)
    
    # Regional baseline rainfall distributions
    region_rain_bases = {
        "North": 45.0,
        "South": 60.0,
        "East": 75.0,
        "West": 50.0,
        "Central": 55.0
    }
    
    all_rows = []
    
    for cfg in commodity_configs:
        commodity = cfg["commodity"]
        category = cfg["category"]
        region = cfg["region"]
        market = cfg["market"]
        base_p = cfg["base_price"]
        msp_base = cfg["msp_base"]
        msp_growth = cfg["msp_growth"]
        peak_m = cfg["season_peak_month"]
        lean_m = cfg["lean_month"]
        seasonal_amp = cfg["seasonal_amplitude"]
        base_arr = cfg["base_arrivals"]
        arr_mult = cfg["arrival_season_mult"]
        
        # Long-term inflation drift (approx 4.2% per year)
        annual_inflation = 0.042
        time_trend = np.linspace(1.0, (1.0 + annual_inflation) ** 6, n_weeks)
        
        # Precompute MSP series by calendar year
        msp_series = []
        for d in dates:
            years_elapsed = d.year - 2020
            if msp_base > 0:
                msp_val = msp_base * ((1.0 + msp_growth) ** years_elapsed)
            else:
                msp_val = 0.0
            msp_series.append(round(msp_val, 2))
            
        # Realistic Autoregressive AR(1) persistent market momentum process (rho = 0.85)
        # Represents persistent supply-chain shocks, storage build-up, and regional trader expectations
        ar_noise = np.zeros(n_weeks)
        for t in range(1, n_weeks):
            ar_noise[t] = 0.84 * ar_noise[t - 1] + np.random.normal(0, 0.018)
            
        # Generate weekly series
        for i, dt in enumerate(dates):
            month = dt.month
            year = dt.year
            
            # Rainfall seasonality (Monsoon peak June-Sept: months 6-9)
            base_rain = region_rain_bases[region]
            monsoon_weight = np.exp(-0.5 * ((month - 7.5) / 1.6) ** 2)
            rainfall = max(0.0, base_rain * (0.2 + 2.8 * monsoon_weight) + np.random.normal(0, 8.0))
            
            # Supply arrivals (peaks at harvest month, drops during lean month)
            dist_from_peak = np.abs(month - peak_m)
            dist_from_peak = min(dist_from_peak, 12 - dist_from_peak)
            arrival_factor = 1.0 + (arr_mult - 1.0) * np.exp(-0.5 * (dist_from_peak / 1.4) ** 2)
            arrival_qty = max(200.0, base_arr * arrival_factor * (1.0 + np.random.normal(0, 0.08)))
            
            # Seasonal price trajectory: inverse of harvest glut, peak during lean demand
            dist_from_lean = np.abs(month - lean_m)
            dist_from_lean = min(dist_from_lean, 12 - dist_from_lean)
            seasonal_price_effect = 1.0 + seasonal_amp * np.exp(-0.5 * (dist_from_lean / 1.6) ** 2) - (seasonal_amp * 0.75) * np.exp(-0.5 * (dist_from_peak / 1.6) ** 2)
            
            # Supply elasticity shock (if arrivals surge above baseline, prices soften)
            supply_elasticity = -0.15 * ((arrival_qty / base_arr) - 1.0)
            
            # Weather anomaly effect (e.g. 2022 heatwave in March/April; 2023 tomato surge; 2024 onion delay)
            weather_shock = 0.0
            if commodity == "Wheat" and year == 2022 and month in [3, 4, 5]:
                weather_shock = 0.14
            elif commodity == "Tomato" and year == 2023 and month in [6, 7, 8]:
                weather_shock = 0.65
            elif commodity == "Onion" and year == 2024 and month in [9, 10, 11]:
                weather_shock = 0.35
                
            # Transport fuel cost contribution
            transport_factor = 1.0 + 0.10 * ((fuel_trend[i] - 100.0) / 100.0)
            
            # Calculate Modal Price incorporating macro trend, seasonal curve, elasticity, shocks, and persistent momentum
            modal_price = (
                base_p * time_trend[i] * seasonal_price_effect *
                (1.0 + supply_elasticity + weather_shock + ar_noise[i]) * transport_factor
            )
            
            # Enforce MSP floor for covered crops
            if msp_series[i] > 0:
                modal_price = max(modal_price, msp_series[i] * np.random.uniform(0.98, 1.25))
                
            # Price boundaries
            price_spread = modal_price * np.random.uniform(0.04, 0.09)
            min_price = max(100.0, modal_price - price_spread)
            max_price = modal_price + price_spread * np.random.uniform(1.0, 1.35)
            
            all_rows.append({
                "date": dt.strftime("%Y-%m-%d"),
                "commodity": commodity,
                "category": category,
                "region": region,
                "market": market,
                "arrival_quantity_tonnes": round(arrival_qty, 2),
                "rainfall_mm": round(rainfall, 2),
                "fuel_price_index": round(fuel_trend[i], 2),
                "msp": msp_series[i],
                "modal_price": round(modal_price, 2),
                "min_price": round(min_price, 2),
                "max_price": round(max_price, 2),
            })
            
    df = pd.DataFrame(all_rows)
    
    # Introduce ~1.2% realistic missing values to validate automated imputation
    mask_miss_arr = np.random.rand(len(df)) < 0.012
    df.loc[mask_miss_arr, "arrival_quantity_tonnes"] = np.nan
    mask_miss_rain = np.random.rand(len(df)) < 0.010
    df.loc[mask_miss_rain, "rainfall_mm"] = np.nan
    
    # Introduce ~0.4% recording error outliers to validate IQR filtering
    outlier_idx = np.random.choice(df.index, size=int(len(df) * 0.004), replace=False)
    for idx in outlier_idx:
        err_type = np.random.choice(["10x", "negative", "huge_arr"])
        if err_type == "10x":
            df.loc[idx, "modal_price"] = df.loc[idx, "modal_price"] * 9.5
            df.loc[idx, "max_price"] = df.loc[idx, "max_price"] * 9.5
        elif err_type == "negative":
            df.loc[idx, "modal_price"] = -1.0 * abs(df.loc[idx, "modal_price"])
        elif err_type == "huge_arr":
            df.loc[idx, "arrival_quantity_tonnes"] = 999999.0
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} commodity records across {df['commodity'].nunique()} crops and {df['market'].nunique()} markets.")
    print(f"Saved raw dataset to: {output_path}")
    return df

if __name__ == "__main__":
    generate_market_data()
