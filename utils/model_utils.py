import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelUtils:
    def __init__(self):
        self.evaluation_history = []
        
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Comprehensive model evaluation"""
        start_time = time.time()
        
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.predict(X_test)
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'evaluation_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        # ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        metrics['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        metrics['pr_curve'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist()
        }
        
        # Store evaluation history
        self.evaluation_history.append(metrics)
        
        return metrics
    
    def cross_validate_model(self, model, X, y, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']):
        """Perform cross-validation on model"""
        cv_results = {}
        
        for score in scoring:
            scores = cross_val_score(model, X, y, cv=cv, scoring=score)
            cv_results[score] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
        
        return cv_results
    
    def hyperparameter_tuning(self, model, param_grid, X_train, y_train, cv=5, 
                            scoring='accuracy', search_type='grid', n_iter=100):
        """Perform hyperparameter tuning"""
        if search_type == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=cv, scoring=scoring, 
                n_jobs=-1, verbose=1, return_train_score=True
            )
        else:  # randomized search
            search = RandomizedSearchCV(
                model, param_grid, cv=cv, scoring=scoring, 
                n_jobs=-1, verbose=1, return_train_score=True,
                n_iter=n_iter, random_state=42
            )
        
        search.fit(X_train, y_train)
        
        results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_estimator': search.best_estimator_,
            'cv_results': search.cv_results_
        }
        
        return results
    
    def model_comparison(self, models, X_train, y_train, X_test, y_test, cv=5):
        """Compare multiple models"""
        comparison_results = {}
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_results = self.cross_validate_model(model, X_train, y_train, cv=cv)
            
            # Test set evaluation
            test_metrics = self.evaluate_model(model, X_test, y_test, model_name)
            
            comparison_results[model_name] = {
                'cross_validation': cv_results,
                'test_metrics': test_metrics,
                'model_object': model
            }
        
        return comparison_results
    
    def feature_selection(self, X, y, method='mutual_info', k=20):
        """Select top k features using specified method"""
        from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, chi2
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'chi2':
            selector = SelectKBest(score_func=chi2, k=k)
        else:
            raise ValueError("Method must be 'mutual_info', 'f_classif', or 'chi2'")
        
        X_selected = selector.fit_transform(X, y)
        selected_features = selector.get_support(indices=True)
        scores = selector.scores_
        
        return {
            'X_selected': X_selected,
            'selected_features': selected_features,
            'scores': scores,
            'selector': selector
        }
    
    def calculate_model_stability(self, model, X, y, n_splits=10):
        """Calculate model stability across different train/test splits"""
        from sklearn.model_selection import ShuffleSplit
        
        cv = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
        
        accuracies = []
        predictions_consistency = []
        
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Fit and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            
            # Store predictions for consistency analysis
            predictions_consistency.append(y_pred)
        
        stability_metrics = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'accuracy_cv': np.std(accuracies) / np.mean(accuracies),  # Coefficient of variation
            'accuracy_range': np.max(accuracies) - np.min(accuracies),
            'individual_accuracies': accuracies
        }
        
        return stability_metrics
    
    def learning_curve_analysis(self, model, X, y, train_sizes=None, cv=5):
        """Analyze learning curves to detect overfitting/underfitting"""
        from sklearn.model_selection import learning_curve
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=cv, 
            scoring='accuracy', n_jobs=-1, random_state=42
        )
        
        learning_curve_data = {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'val_scores_mean': np.mean(val_scores, axis=1),
            'val_scores_std': np.std(val_scores, axis=1),
            'overfitting_gap': np.mean(train_scores, axis=1) - np.mean(val_scores, axis=1)
        }
        
        return learning_curve_data
    
    def validation_curve_analysis(self, model, X, y, param_name, param_range, cv=5):
        """Analyze validation curves for hyperparameter tuning"""
        from sklearn.model_selection import validation_curve
        
        train_scores, val_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring='accuracy', n_jobs=-1
        )
        
        validation_curve_data = {
            'param_range': param_range,
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'val_scores_mean': np.mean(val_scores, axis=1),
            'val_scores_std': np.std(val_scores, axis=1)
        }
        
        return validation_curve_data
    
    def ensemble_predictions(self, models, X, weights=None):
        """Combine predictions from multiple models"""
        if weights is None:
            weights = [1/len(models)] * len(models)
        
        predictions = []
        for model in models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        return ensemble_pred
    
    def calibration_analysis(self, model, X, y, n_bins=10):
        """Analyze model calibration"""
        from sklearn.calibration import calibration_curve
        
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)[:, 1]
        else:
            y_prob = model.predict(X)
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y, y_prob, n_bins=n_bins
        )
        
        calibration_data = {
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value,
            'n_bins': n_bins
        }
        
        return calibration_data
    
    def calculate_prediction_intervals(self, models, X, confidence=0.95):
        """Calculate prediction intervals using ensemble of models"""
        all_predictions = []
        
        for model in models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        intervals = {
            'mean': np.mean(all_predictions, axis=0),
            'std': np.std(all_predictions, axis=0),
            'lower': np.percentile(all_predictions, lower_percentile, axis=0),
            'upper': np.percentile(all_predictions, upper_percentile, axis=0),
            'confidence': confidence
        }
        
        return intervals
    
    def model_interpretability(self, model, X, feature_names, method='permutation'):
        """Analyze model interpretability"""
        if method == 'permutation':
            from sklearn.inspection import permutation_importance
            
            if hasattr(model, 'predict_proba'):
                scoring = 'roc_auc'
            else:
                scoring = 'accuracy'
            
            perm_importance = permutation_importance(
                model, X, y, scoring=scoring, n_repeats=10, random_state=42
            )
            
            importance_data = pd.DataFrame({
                'feature': feature_names,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            return importance_data
        
        elif method == 'shap' and hasattr(model, 'predict_proba'):
            try:
                import shap
                explainer = shap.Explainer(model, X)
                shap_values = explainer(X)
                
                shap_data = {
                    'shap_values': shap_values.values,
                    'base_values': shap_values.base_values,
                    'feature_names': feature_names
                }
                
                return shap_data
            except ImportError:
                print("SHAP not installed. Using permutation importance instead.")
                return self.model_interpretability(model, X, feature_names, method='permutation')
    
    def save_model_artifacts(self, model, model_name, artifacts_dir="model_artifacts"):
        """Save model and related artifacts"""
        if not os.path.exists(artifacts_dir):
            os.makedirs(artifacts_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(artifacts_dir, f"{model_name}_{timestamp}.pkl")
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save evaluation history
        history_path = os.path.join(artifacts_dir, f"{model_name}_evaluation_history_{timestamp}.json")
        import json
        with open(history_path, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2)
        
        return {
            'model_path': model_path,
            'history_path': history_path,
            'timestamp': timestamp
        }
    
    def load_model_artifacts(self, model_path):
        """Load saved model artifacts"""
        model = joblib.load(model_path)
        
        # Try to load evaluation history
        history_path = model_path.replace('.pkl', '_evaluation_history.json')
        if os.path.exists(history_path):
            import json
            with open(history_path, 'r') as f:
                evaluation_history = json.load(f)
        else:
            evaluation_history = []
        
        return model, evaluation_history
    
    def generate_model_report(self, model_results, output_path="model_report.html"):
        """Generate comprehensive model evaluation report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cricket Century Prediction Model Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                .section {{ margin: 20px 0; border: 1px solid #ddd; padding: 15px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Cricket Century Prediction Model Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Model Performance Summary</h2>
                <table>
                    <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>ROC-AUC</th></tr>
        """
        
        for model_name, results in model_results.items():
            metrics = results['test_metrics']
            html_content += f"""
                    <tr>
                        <td>{model_name}</td>
                        <td>{metrics['accuracy']:.4f}</td>
                        <td>{metrics['precision']:.4f}</td>
                        <td>{metrics['recall']:.4f}</td>
                        <td>{metrics['f1_score']:.4f}</td>
                        <td>{metrics['roc_auc']:.4f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Evaluation History</h2>
                <p>This section contains the evaluation history of all models tested.</p>
            </div>
            
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path
    
    def get_model_summary(self, model):
        """Get a summary of model parameters and architecture"""
        summary = {
            'model_type': type(model).__name__,
            'parameters': {},
            'feature_importance': None
        }
        
        # Get model parameters
        if hasattr(model, 'get_params'):
            summary['parameters'] = model.get_params()
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            summary['feature_importance'] = model.feature_importances_.tolist()
        elif hasattr(model, 'coef_'):
            summary['feature_importance'] = model.coef_.tolist()
        
        return summary
