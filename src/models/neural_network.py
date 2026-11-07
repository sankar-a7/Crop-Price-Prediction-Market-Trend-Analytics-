"""
Deep Neural Network (DNN) Regressor Architecture for Crop Price Forecasting.
Implemented in PyTorch with Batch Normalization, Dropout, and Adaptive Learning Rate Scheduling.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class CropPriceDNN(nn.Module):
    """
    Multi-layer Deep Perceptron architecture for non-linear price trajectory forecasting.
    """

    def __init__(self, input_dim: int, hidden_dim1: int = 128, hidden_dim2: int = 64, hidden_dim3: int = 32):
        super(CropPriceDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            
            nn.Linear(hidden_dim3, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepNeuralNetworkPriceForecaster:
    """
    PyTorch-based Deep Neural Network regressor wrapper with automated feature scaling
    and PyTorch execution lifecycle.
    """

    def __init__(
        self,
        epochs: int = 60,
        batch_size: int = 64,
        learning_rate: float = 0.005,
        device: Optional[str] = None
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model: Optional[CropPriceDNN] = None
        self.feature_names_: List[str] = []
        self.training_losses_: List[float] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = list(X.columns)
        input_dim = len(self.feature_names_)
        
        # Scale X and y for stable neural convergence
        X_scaled = self.scaler_x.fit_transform(X.values)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        # Prepare Tensors
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
        tensor_y = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(1)
        
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = CropPriceDNN(input_dim).to(self.device)
        criterion = nn.SmoothL1Loss()  # Huber loss for outlier robustness
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        
        self.model.train()
        self.training_losses_ = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)
                
            epoch_loss /= len(dataset)
            self.training_losses_.append(epoch_loss)
            scheduler.step(epoch_loss)
            
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        self.model.eval()
        X_scaled = self.scaler_x.transform(X.values)
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds_scaled = self.model(tensor_x).cpu().numpy()
            
        preds = self.scaler_y.inverse_transform(preds_scaled).flatten()
        return preds

    def save(self, filepath: str = "models/deep_neural_network.pt"):
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        payload = {
            "state_dict": self.model.state_dict(),
            "scaler_x": self.scaler_x,
            "scaler_y": self.scaler_y,
            "feature_names": self.feature_names_,
            "input_dim": len(self.feature_names_),
            "epochs": self.epochs,
            "training_losses": self.training_losses_
        }
        torch.save(payload, filepath)

    @classmethod
    def load(cls, filepath: str = "models/deep_neural_network.pt") -> "DeepNeuralNetworkPriceForecaster":
        payload = torch.load(filepath, map_location="cpu", weights_only=False)
        instance = cls(epochs=payload.get("epochs", 50))
        instance.feature_names_ = payload["feature_names"]
        instance.scaler_x = payload["scaler_x"]
        instance.scaler_y = payload["scaler_y"]
        instance.training_losses_ = payload.get("training_losses", [])
        
        input_dim = payload["input_dim"]
        instance.model = CropPriceDNN(input_dim)
        instance.model.load_state_dict(payload["state_dict"])
        instance.model.to(instance.device)
        instance.model.eval()
        return instance
