"""
Gradient Boosted Decision Tree (XGBoost / Gradient Boosting) Architecture.
Provides high-performance gradient boosting forecasting with automated fallback.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import joblib

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class GradientBoostedPriceForecaster:
    """
    Gradient Boosted Regression Tree architecture for agricultural market trend analytics.
    Uses XGBoost when available or Scikit-learn Gradient Boosting with unified interface.
    """

    def __init__(
        self,
        learning_rate: float = 0.06,
        n_estimators: int = 150,
        max_depth: int = 5,
        subsample: float = 0.85,
        random_state: int = 42
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        self.is_xgboost = XGBOOST_AVAILABLE
        
        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBRegressor(
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                max_depth=max_depth,
                subsample=subsample,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            self.model = GradientBoostingRegressor(
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                max_depth=max_depth,
                subsample=subsample,
                random_state=random_state
            )
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = list(X.columns)
        # Ensure finite numeric matrix
        X_mat = np.nan_to_num(X.values, nan=0.0, posinf=1e6, neginf=-1e6)
        self.model.fit(X_mat, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_mat = np.nan_to_num(X.values, nan=0.0, posinf=1e6, neginf=-1e6)
        return self.model.predict(X_mat)

    def get_feature_importances(self, top_n: int = 15) -> pd.DataFrame:
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            importances = np.ones(len(self.feature_names_)) / len(self.feature_names_)
            
        df_imp = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": importances
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        return df_imp.head(top_n)

    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, n_iter: int = 6):
        param_dist = {
            "n_estimators": [100, 150, 180],
            "learning_rate": [0.04, 0.06, 0.1],
            "max_depth": [4, 5, 6],
            "subsample": [0.8, 0.9, 1.0]
        }
        tscv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(
            self.model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            random_state=42,
            n_jobs=-1
        )
        search.fit(np.nan_to_num(X.values, nan=0.0), y.values)
        self.model = search.best_estimator_
        self.feature_names_ = list(X.columns)
        return search.best_params_

    def save(self, filepath: str = "models/xgboost_regressor.joblib"):
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names_,
            "is_xgboost": self.is_xgboost
        }, filepath)

    @classmethod
    def load(cls, filepath: str = "models/xgboost_regressor.joblib") -> "GradientBoostedPriceForecaster":
        data = joblib.load(filepath)
        instance = cls()
        instance.model = data["model"]
        instance.feature_names_ = data["feature_names"]
        instance.is_xgboost = data.get("is_xgboost", False)
        return instance
