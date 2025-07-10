
    
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib
import os

class TensorFlowModel:
    def __init__(self, input_dim=50):
        self.input_dim = input_dim
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.use_sklearn_fallback = not TF_AVAILABLE
        
    def build_model(self):
        """Build the TensorFlow/Keras model architecture or sklearn fallback"""
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
            model = keras.Sequential([
                # Input layer
                layers.Dense(128, activation='relu', input_shape=(self.input_dim,)),
                layers.Dropout(0.3),
                
                # Hidden layers
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.1),
                
                # Output layer
                layers.Dense(1, activation='sigmoid')
            ])
            
            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            self.model = model
            return model
    
    def preprocess_data(self, X, y=None, fit_scaler=False):
        """Preprocess the input data"""
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """Train the TensorFlow model or sklearn fallback"""
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
            # Split data for TensorFlow
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=validation_split, random_state=42
            )
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                ),
                keras.callbacks.ModelCheckpoint(
                    'best_tf_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            self.is_trained = True
            return history
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled, _ = self.preprocess_data(X)
        
        if self.use_sklearn_fallback:
            predictions = self.model.predict_proba(X_scaled)[:, 1]
            return predictions
        else:
            predictions = self.model.predict(X_scaled)
            return predictions.flatten()
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.predict(X)
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        X_scaled, y = self.preprocess_data(X, y)
        
        if self.use_sklearn_fallback:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            predictions = self.model.predict(X_scaled)
            pred_proba = self.model.predict_proba(X_scaled)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y, predictions),
                'precision': precision_score(y, predictions, average='binary'),
                'recall': recall_score(y, predictions, average='binary'),
                'f1_score': f1_score(y, predictions, average='binary')
            }
            return metrics
        else:
            results = self.model.evaluate(X_scaled, y, verbose=0)
            
            metrics = {}
            for i, metric in enumerate(self.model.metrics_names):
                metrics[metric] = results[i]
            
            return metrics
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is not None:
            if self.use_sklearn_fallback:
                joblib.dump(self.model, f"{filepath}_sklearn.pkl")
            else:
                self.model.save(f"{filepath}.h5")
            joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
            joblib.dump(self.use_sklearn_fallback, f"{filepath}_fallback.pkl")
    
    def load_model(self, filepath):
        """Load a saved model"""
        try:
            self.use_sklearn_fallback = joblib.load(f"{filepath}_fallback.pkl")
        except:
            self.use_sklearn_fallback = not TF_AVAILABLE
            
        if self.use_sklearn_fallback:
            self.model = joblib.load(f"{filepath}_sklearn.pkl")
        else:
            self.model = keras.models.load_model(f"{filepath}.h5")
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        self.is_trained = True
    
    def get_feature_importance(self, X, y, feature_names):
        """Get feature importance using permutation importance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        X_scaled, _ = self.preprocess_data(X)
        base_score = self.model.evaluate(X_scaled, y, verbose=0)[1]  # accuracy
        
        importance_scores = []
        
        for i in range(X_scaled.shape[1]):
            # Create permuted version
            X_permuted = X_scaled.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            
            # Get score with permuted feature
            permuted_score = self.model.evaluate(X_permuted, y, verbose=0)[1]
            
            # Calculate importance as difference in performance
            importance = base_score - permuted_score
            importance_scores.append(importance)
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is not None:
            return self.model.summary()
        else:
            return "Model not built yet"
