"""
Random Forest Regressor Architecture for Crop Price Forecasting.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import joblib


class RandomForestPriceForecaster:
    """
    Random Forest ensemble regression model for commodity modal price forecasting.
    Optimizes split criteria and feature subsets to capture nonlinear agro-climatic interactions.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: Optional[int] = 16,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        random_state: int = 42
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=random_state
        )
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importances(self, top_n: int = 15) -> pd.DataFrame:
        """
        Extract normalized MDI feature importance scores.
        """
        importances = self.model.feature_importances_
        df_imp = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": importances
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        return df_imp.head(top_n)

    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, n_iter: int = 8):
        """
        Tune hyperparameters using Temporal Cross-Validation.
        """
        param_dist = {
            "n_estimators": [100, 150, 200],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 4, 8],
            "min_samples_leaf": [1, 2, 4]
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
        search.fit(X, y)
        self.model = search.best_estimator_
        self.feature_names_ = list(X.columns)
        return search.best_params_

    def save(self, filepath: str = "models/random_forest.joblib"):
        joblib.dump({"model": self.model, "feature_names": self.feature_names_}, filepath)

    @classmethod
    def load(cls, filepath: str = "models/random_forest.joblib") -> "RandomForestPriceForecaster":
        data = joblib.load(filepath)
        instance = cls()
        instance.model = data["model"]
        instance.feature_names_ = data["feature_names"]
        return instance

