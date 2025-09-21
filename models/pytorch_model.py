# models/pytorch_model.py

# =========================================================================
# FIX: Add the try/except block to handle the PyTorch import and define TORCH_AVAILABLE
# =========================================================================
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not found. Using scikit-learn fallback.")

# =========================================================================
# All other imports should go below the PyTorch check
# =========================================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib

# The rest of your code remains the same
if TORCH_AVAILABLE:
    class CricketCenturyNet(nn.Module):
        def __init__(self, input_dim=50):
            super(CricketCenturyNet, self).__init__()

            self.network = nn.Sequential(
                # Input layer
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),

                # Hidden layers
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.4),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.1),

                # Output layer
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.network(x)
else:
    CricketCenturyNet = None

class PyTorchModel:
    def __init__(self, input_dim=50):
        self.input_dim = input_dim
        self.model = None
        self.scaler = StandardScaler()
        self.use_sklearn_fallback = not TORCH_AVAILABLE
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = None
        self.is_trained = False

    def build_model(self):
        """Build the PyTorch model or sklearn fallback"""
        if self.use_sklearn_fallback:
            # Use sklearn MLPClassifier as fallback
            self.model = MLPClassifier(
                hidden_layer_sizes=(128, 256, 128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=10
            )
            return self.model
        else:
            self.model = CricketCenturyNet(self.input_dim).to(self.device)
            return self.model

    def preprocess_data(self, X, y=None, fit_scaler=False):
        """Preprocess the input data"""
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled, y

    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32, learning_rate=0.001):
        """Train the PyTorch model or sklearn fallback"""
        # Preprocess data
        X_scaled, y = self.preprocess_data(X, y, fit_scaler=True)

        # Build model if not already built
        if self.model is None:
            self.build_model()

        if self.use_sklearn_fallback:
            # Train sklearn model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            return {"sklearn_training": "completed"}
        else:
            # This is the PyTorch training block, which was missing
            from torch.utils.data import DataLoader, TensorDataset
            
            # Convert to tensors
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)
            
            # Create dataset and dataloader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Train loop
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.BCELoss()

            for epoch in range(epochs):
                for inputs, labels in dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
            self.is_trained = True
            return {"pytorch_training": "completed"}

    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_scaled, _ = self.preprocess_data(X)

        if self.use_sklearn_fallback:
            predictions = self.model.predict_proba(X_scaled)[:, 1]
            return predictions
        else:
            # PyTorch prediction logic
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
                outputs = self.model(X_tensor).cpu().numpy().flatten()
            return outputs

    def predict_proba(self, X):
        """Get prediction probabilities"""
        # In this implementation, predict already returns probabilities, so this is a passthrough
        return self.predict(X)

    def evaluate(self, X, y):
        """Evaluate model performance"""
        X_scaled, y = self.preprocess_data(X, y)

        if self.use_sklearn_fallback:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            predictions = self.model.predict(X_scaled)

            metrics = {
                'accuracy': accuracy_score(y, predictions),
                'precision': precision_score(y, predictions, average='binary'),
                'recall': recall_score(y, predictions, average='binary'),
                'f1_score': f1_score(y, predictions, average='binary')
            }
            return metrics
        else:
            # PyTorch evaluation logic
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
                outputs = self.model(X_tensor).cpu().numpy()
            
            predictions = (outputs > 0.5).astype(int).flatten()
            
            metrics = {
                'accuracy': accuracy_score(y, predictions),
                'precision': precision_score(y, predictions, average='binary'),
                'recall': recall_score(y, predictions, average='binary'),
                'f1_score': f1_score(y, predictions, average='binary')
            }
            return metrics

    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is not None:
            if self.use_sklearn_fallback:
                joblib.dump(self.model, f"{filepath}_sklearn.pkl")
            else:
                torch.save(self.model.state_dict(), f"{filepath}_pytorch.pth")
            joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
            joblib.dump(self.use_sklearn_fallback, f"{filepath}_fallback.pkl")
        else:
            raise RuntimeError("Model has not been trained yet. Cannot save.")

    def load_model(self, filepath):
        """Load a saved model"""
        try:
            self.use_sklearn_fallback = joblib.load(f"{filepath}_fallback.pkl")
        except FileNotFoundError:
            self.use_sklearn_fallback = not TORCH_AVAILABLE

        if self.use_sklearn_fallback:
            self.model = joblib.load(f"{filepath}_sklearn.pkl")
        else:
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch is not available, but the saved model requires it.")
            
            self.build_model()
            self.model.load_state_dict(torch.load(f"{filepath}_pytorch.pth"))
            self.model.eval()
            
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        self.is_trained = True

    def get_feature_importance(self, X, y, feature_names):
        """Get feature importance using permutation importance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")

        if self.use_sklearn_fallback:
            from sklearn.inspection import permutation_importance
            X_scaled, _ = self.preprocess_data(X)
            result = permutation_importance(self.model, X_scaled, y, 
                                          n_repeats=10, random_state=42)

            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': result.importances_mean
            }).sort_values('importance', ascending=False)

            return feature_importance
        else:
            raise RuntimeError("Permutation importance for PyTorch is not implemented in this code.")

