import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from .tensorflow_model import TensorFlowModel
from .pytorch_model import PyTorchModel

class EnsembleModel:
    def __init__(self, input_dim=50):
        self.input_dim = input_dim
        self.tf_model = TensorFlowModel(input_dim)
        self.pytorch_model = PyTorchModel(input_dim)
        self.sklearn_models = None
        self.ensemble_weights = None
        self.is_trained = False
        
    def build_sklearn_models(self):
        """Build traditional ML models for ensemble"""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }
        
        self.sklearn_models = models
        return models
    
    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """Train all models in the ensemble"""
        print("Training TensorFlow model...")
        tf_history = self.tf_model.train(X, y, validation_split, epochs, batch_size)
        
        print("Training PyTorch model...")
        pytorch_history = self.pytorch_model.train(X, y, validation_split, epochs, batch_size)
        
        print("Training sklearn models...")
        if self.sklearn_models is None:
            self.build_sklearn_models()
        
        # Preprocess data for sklearn models
        X_scaled, _ = self.tf_model.preprocess_data(X, y)
        
        # Train sklearn models
        sklearn_histories = {}
        for name, model in self.sklearn_models.items():
            print(f"Training {name}...")
            model.fit(X_scaled, y)
            sklearn_histories[name] = "Trained successfully"
        
        # Calculate ensemble weights based on validation performance
        self._calculate_ensemble_weights(X, y)
        
        self.is_trained = True
        
        return {
            'tensorflow': tf_history,
            'pytorch': pytorch_history,
            'sklearn': sklearn_histories
        }
    
    def _calculate_ensemble_weights(self, X, y):
        """Calculate weights for ensemble based on model performance"""
        from sklearn.model_selection import cross_val_score
        
        # Get predictions from deep learning models
        tf_predictions = self.tf_model.predict(X)
        pytorch_predictions = self.pytorch_model.predict(X)
        
        # Preprocess data for sklearn models
        X_scaled, _ = self.tf_model.preprocess_data(X, y)
        
        # Calculate cross-validation scores
        tf_score = accuracy_score(y, (tf_predictions > 0.5).astype(int))
        pytorch_score = accuracy_score(y, (pytorch_predictions > 0.5).astype(int))
        
        sklearn_scores = {}
        for name, model in self.sklearn_models.items():
            scores = cross_val_score(model, X_scaled, y, cv=5)
            sklearn_scores[name] = scores.mean()
        
        # Calculate weights (higher score = higher weight)
        all_scores = {
            'tensorflow': tf_score,
            'pytorch': pytorch_score,
            **sklearn_scores
        }
        
        # Normalize weights
        total_score = sum(all_scores.values())
        self.ensemble_weights = {
            model: score / total_score for model, score in all_scores.items()
        }
        
        print("Ensemble weights:", self.ensemble_weights)
    
    def predict(self, X):
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        # Get predictions from all models
        tf_predictions = self.tf_model.predict(X)
        pytorch_predictions = self.pytorch_model.predict(X)
        
        # Preprocess data for sklearn models
        X_scaled, _ = self.tf_model.preprocess_data(X)
        
        sklearn_predictions = {}
        for name, model in self.sklearn_models.items():
            sklearn_predictions[name] = model.predict_proba(X_scaled)[:, 1]
        
        # Weighted ensemble prediction
        ensemble_prediction = np.zeros(len(X))
        
        ensemble_prediction += self.ensemble_weights['tensorflow'] * tf_predictions
        ensemble_prediction += self.ensemble_weights['pytorch'] * pytorch_predictions
        
        for name, predictions in sklearn_predictions.items():
            ensemble_prediction += self.ensemble_weights[name] * predictions
        
        return ensemble_prediction
    
    def predict_proba(self, X):
        """Get ensemble prediction probabilities"""
        return self.predict(X)
    
    def predict_with_uncertainty(self, X):
        """Get predictions with uncertainty estimates"""
        # Get predictions from all models
        tf_predictions = self.tf_model.predict(X)
        pytorch_predictions = self.pytorch_model.predict(X)
        
        X_scaled, _ = self.tf_model.preprocess_data(X)
        
        all_predictions = [tf_predictions, pytorch_predictions]
        
        for name, model in self.sklearn_models.items():
            sklearn_pred = model.predict_proba(X_scaled)[:, 1]
            all_predictions.append(sklearn_pred)
        
        # Calculate mean and std
        predictions_array = np.array(all_predictions)
        mean_prediction = np.mean(predictions_array, axis=0)
        std_prediction = np.std(predictions_array, axis=0)
        
        return mean_prediction, std_prediction
    
    def evaluate(self, X, y):
        """Evaluate ensemble performance"""
        predictions = self.predict(X)
        binary_predictions = (predictions > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y, binary_predictions),
            'precision': precision_score(y, binary_predictions),
            'recall': recall_score(y, binary_predictions),
            'f1_score': f1_score(y, binary_predictions)
        }
        
        return metrics
    
    def get_model_predictions(self, X):
        """Get individual model predictions for analysis"""
        tf_predictions = self.tf_model.predict(X)
        pytorch_predictions = self.pytorch_model.predict(X)
        
        X_scaled, _ = self.tf_model.preprocess_data(X)
        
        sklearn_predictions = {}
        for name, model in self.sklearn_models.items():
            sklearn_predictions[name] = model.predict_proba(X_scaled)[:, 1]
        
        return {
            'tensorflow': tf_predictions,
            'pytorch': pytorch_predictions,
            **sklearn_predictions
        }
    
    def save_ensemble(self, filepath):
        """Save the entire ensemble"""
        self.tf_model.save_model(f"{filepath}_tensorflow")
        self.pytorch_model.save_model(f"{filepath}_pytorch")
        
        # Save sklearn models
        for name, model in self.sklearn_models.items():
            joblib.dump(model, f"{filepath}_{name}.pkl")
        
        # Save ensemble weights
        joblib.dump(self.ensemble_weights, f"{filepath}_weights.pkl")
    
    def load_ensemble(self, filepath):
        """Load the entire ensemble"""
        self.tf_model.load_model(f"{filepath}_tensorflow")
        self.pytorch_model.load_model(f"{filepath}_pytorch")
        
        # Load sklearn models
        self.build_sklearn_models()
        for name in self.sklearn_models.keys():
            self.sklearn_models[name] = joblib.load(f"{filepath}_{name}.pkl")
        
        # Load ensemble weights
        self.ensemble_weights = joblib.load(f"{filepath}_weights.pkl")
        self.is_trained = True
    
    def get_feature_importance(self, X, y, feature_names):
        """Get ensemble feature importance"""
        # Get feature importance from each model
        tf_importance = self.tf_model.get_feature_importance(X, y, feature_names)
        pytorch_importance = self.pytorch_model.get_feature_importance(X, y, feature_names)
        
        # Get feature importance from sklearn models
        X_scaled, _ = self.tf_model.preprocess_data(X)
        
        rf_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.sklearn_models['random_forest'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Weight and combine importances
        tf_weight = self.ensemble_weights['tensorflow']
        pytorch_weight = self.ensemble_weights['pytorch']
        rf_weight = self.ensemble_weights['random_forest']
        
        # Create combined importance
        combined_importance = pd.DataFrame({'feature': feature_names})
        combined_importance['importance'] = (
            tf_weight * tf_importance.set_index('feature').loc[feature_names, 'importance'].values +
            pytorch_weight * pytorch_importance.set_index('feature').loc[feature_names, 'importance'].values +
            rf_weight * rf_importance.set_index('feature').loc[feature_names, 'importance'].values
        )
        
        return combined_importance.sort_values('importance', ascending=False)
