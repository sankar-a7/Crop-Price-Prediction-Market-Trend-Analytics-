---
marp: true
theme: default
paginate: true
header: "Crop Price Prediction & Market Trend Analytics | Major Project 2024–2025"
footer: "End-to-End Machine Learning Forecasting System"
style: |
  section {
    background-color: #fdfdfd;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  }
  h1 {
    color: #1b4332;
  }
  h2 {
    color: #2d6a4f;
  }
  .highlight {
    color: #d90429;
    font-weight: bold;
  }
  .badge {
    background-color: #d8f3dc;
    color: #1b4332;
    padding: 4px 8px;
    border-radius: 4px;
    font-weight: bold;
  }
---

# 🌾 Crop Price Prediction & Market Trend Analytics
### End-to-End Machine Learning Forecasting & Decision Support System (2024 – 2025)

**Domain:** Applied Machine Learning • Agricultural Econometrics • Time-Series  
**Tech Stack:** Python, Scikit-learn, Random Forest, XGBoost, PyTorch, Pandas, NumPy, Streamlit, Matplotlib, Git  

---

## 1. Problem Statement & Motivation

- **High Price Volatility:** Indian APMC mandis experience acute, unpredictable price swings driven by monsoon variability, harvest gluts, and fuel inflation.
- **Distress Sales:** Smallholder farmers lack forward visibility and suffer price crashes during peak arrival weeks.
- **Limitations of Classical Econometrics:** Traditional ARIMA/SARIMAX models assume linear stationarity and fail during abrupt supply shocks (e.g. unseasonal rain or export bans).
- **Project Goal:** Design, benchmark, and deploy an end-to-end ML forecasting pipeline achieving **>90% price trend accuracy**.

---

## 2. System Architecture & Workflow

```
[Raw APMC Market Data (2020-2025)]
               │
               ▼
[Automated Preprocessing Engine]
 ├── Hierarchical Time-Series Imputation
 └── Domain-Tuned IQR Outlier Removal (k=2.5)
               │
               ▼
[Domain Feature Engineering]
 ├── Cyclical Calendar Harmonics (sin, cos)
 ├── Agro-Climatic Seasons (Kharif, Rabi, Zaid)
 ├── Autoregressive Lags (1w to 52w) & Volatilities
 └── Supply Arrival Elasticity & MSP Ratios
               │
               ▼
[Multi-Model Benchmarking (Chronological 2025 Holdout)]
 ├── Random Forest Regressor
 ├── XGBoost / Gradient Boosting
 └── Deep Neural Network (PyTorch 4-Layer MLP)
               │
               ▼
[Deployment: Streamlit Dashboard + Flask REST API]
```

---

## 3. Data Preprocessing & Outlier Strategy

- **Dataset Scale:** 4,000+ historical weekly market records across 11 crop varieties and 13 primary APMC mandis (2020–2025).
- **Hierarchical Imputation:**
  1. Temporal linear interpolation within each commodity-mandi series.
  2. Forward & backward fills within regional crop categories.
  3. Grouped regional median imputation fallback.
- **Domain-Aware Outlier Cleaning:**
  - Standard statistical $3\sigma$ filters inadvertently truncate true seasonal price peaks (e.g., lean season onion/tomato spikes).
  - Implemented **commodity-specific IQR bounding ($k=2.5$)**, targeting only recording typos and sensor errors without discarding legitimate seasonal volatility.

---

## 4. Domain-Specific Feature Engineering

- **Cyclical Harmonics (Preserving Circular Continuity):**
  $$\text{month\_sin} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$
- **Autoregressive Price Signals:**
  - Lags at $t-1\text{w}, t-2\text{w}, t-4\text{w}, t-8\text{w}, t-12\text{w}, t-52\text{w}$ (year-over-year baseline).
- **Rolling Volatility & Momentum:**
  - 4-week and 12-week moving averages and rolling standard deviations.
- **Supply-Demand Elasticity Ratios:**
  $$\text{Supply Ratio}_t = \frac{\text{Arrival}_t}{\overline{\text{Arrival}}_{4\text{w}} + \epsilon}$$

---

## 5. Chronological Validation & Model Architectures

### Out-of-Time Temporal Split:
- **Training Set (2020–2024):** 3,379 observations across 5 years.
- **Holdout Test Set (2025):** 674 observations (simulating true live deployment).
- **Zero Lookahead Leakage:** Moving averages and scalers strictly use past records.

### Benchmarked Architectures:
1. **Random Forest Regressor:** 160 estimators, max depth 16, MDI split criteria.
2. **XGBoost / Gradient Boosting:** Learning rate 0.06, depth 5, histogram boosting.
3. **Deep Neural Network (PyTorch):** 4-layer feedforward architecture with Batch Normalization, ReLU, Dropout (0.15), and AdamW optimizer.

---

## 6. Experimental Results & Leaderboard

Evaluated strictly on the unseen **2025 Out-of-Time Test Set**:

| Model Architecture | RMSE (₹/qtl) | MAE (₹/qtl) | MAPE (%) | $R^2$ Score | Directional Trend Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **359.36** | 244.43 | **5.49%** | **0.9804** | **91.25%** |
| **XGBoost / GradBoost** | 366.34 | **243.75** | 5.51% | 0.9796 | **91.39%** |
| **Deep Neural Net (PyTorch)**| 391.30 | 273.03 | 5.82% | 0.9768 | 89.91% |

<span class="badge">Result</span> Achieved **91.17% average directional price trend accuracy** across regional crop categories.

---

## 7. Regional Category Performance Breakdown

| Region | Crop Category | Benchmark Mandi | Trend Forecasting Accuracy |
| :--- | :--- | :--- | :---: |
| **North** | Cereals (Wheat) | Khanna (Punjab) | **94.23%** |
| **North** | Oilseeds (Mustard) | Jaipur (Rajasthan) | **94.23%** |
| **North** | Vegetables (Potato) | Agra (UP) | **92.31%** |
| **South** | Cereals (Rice & Maize) | Nizamabad & Davanagere | **92.23%** |
| **South** | Vegetables (Tomato) | Kolar (Karnataka) | **94.23%** |
| **West** | Cash Crops (Cotton) | Rajkot (Gujarat) | **92.31%** |
| **West** | Vegetables (Onion) | Lasalgaon (Maharashtra) | **94.23%** |
| **East** | Cereals (Rice) | Burdwan (West Bengal) | **92.16%** |

---

## 8. Interactive Streamlit Dashboard

- **Module 1: Seasonal Trajectories & Market Trends**
  - Dual-axis price vs. arrival volume interactive curves with Government MSP floor.
- **Module 2: What-If Scenario Forecasting Simulator**
  - Live simulation of arrival shocks ($-50\%$ to $+80\%$), rainfall anomalies, and fuel index changes.
- **Module 3: Model Benchmarking & Parity Plots**
  - Out-of-time parity correlation ($y = x$) and top 10 feature importance rankings.
- **Module 4: Regional Market Spread & Arbitrage**
  - Identifies inter-mandi price divergences to highlight transport friction and arbitrage.

---

## 9. Production Engineering, API & CI/CD

- **Production REST API (`api.py`):**
  - Low-latency Flask endpoints (`/health`, `/commodities`, `/benchmarks`, `/forecast`, `/scenario`).
- **Explainability Engine (`src/explainability.py`):**
  - Decomposes forecasts into marginal domain contributions: Lags, Seasonality, Arrival Elasticity, and Macro Fuel pressures.
- **Enterprise CI/CD & Containers:**
  - Automated GitHub Actions workflow testing on Python 3.10, 3.11, and 3.12.
  - Multi-stage `Dockerfile` and `docker-compose.yml`.

---

## 10. Summary & Future Scope

### Project Achievements:
- Ingested & preprocessed 6 years of historical APMC records.
- Benchmarked Random Forest, XGBoost, and PyTorch Neural Networks.
- **91.2% Directional Trend Forecasting Accuracy** across regional crop categories.
- Real-time interactive dashboard deployed on Streamlit.
- 100% reproducible pipeline maintained via Git.

### Future Enhancements:
- Ingestion of satellite remote sensing data (Sentinel-2 / MODIS NDVI indices).
- Multi-lingual automated WhatsApp / IVR voice alerts for local farmers.

---

# Thank You!
### Questions & Technical Discussion Welcome
