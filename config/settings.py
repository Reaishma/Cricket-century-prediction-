
import os
from datetime import datetime

class Settings:
    """Configuration settings for the Cricket Century Prediction Platform"""
    
    def __init__(self):
        # API Configuration
        self.API_KEY = os.getenv('CRICKET_API_KEY', 'demo_key')
        self.WEATHER_API_KEY = os.getenv('WEATHER_API_KEY', 'demo_weather_key')
        
        # Model Configuration
        self.MODEL_CONFIG = {
            'tensorflow': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'dropout_rate': 0.3,
                'hidden_layers': [128, 256, 128, 64, 32],
                'activation': 'relu',
                'optimizer': 'adam',
                'loss': 'binary_crossentropy'
            },
            'pytorch': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'dropout_rate': 0.3,
                'hidden_layers': [128, 256, 128, 64, 32],
                'activation': 'relu',
                'optimizer': 'adam',
                'scheduler_patience': 5,
                'early_stopping_patience': 10
            },
            'ensemble': {
                'voting_type': 'soft',
                'weights': None,  # Auto-calculated based on performance
                'cross_validation': 5
            }
        }
        
        # Data Configuration
        self.DATA_CONFIG = {
            'train_test_split': 0.8,
            'validation_split': 0.2,
            'random_state': 42,
            'min_matches_threshold': 10,
            'feature_selection_k': 25,
            'outlier_threshold': 3.0,
            'missing_value_threshold': 0.1
        }
        
        # Feature Engineering Configuration
        self.FEATURE_CONFIG = {
            'rolling_windows': [5, 10, 20],
            'form_window': 10,
            'consistency_window': 10,
            'momentum_window': 5,
            'scale_features': True,
            'create_interaction_features': True,
            'create_polynomial_features': False,
            'polynomial_degree': 2
        }
        
        # API Endpoints
        self.API_ENDPOINTS = {
            'base_url': 'https://api.cricapi.com/v1',
            'current_matches': '/currentMatches',
            'match_details': '/matches/{match_id}',
            'player_stats': '/players/{player_id}',
            'live_score': '/matches/{match_id}/live',
            'team_stats': '/teams',
            'weather_base': 'https://api.openweathermap.org/data/2.5'
        }
        
        # Caching Configuration
        self.CACHE_CONFIG = {
            'enable_caching': True,
            'cache_ttl': 3600,  # 1 hour
            'cache_size': 1000,
            'cache_type': 'memory'  # or 'redis' for production
        }
        
        # Logging Configuration
        self.LOG_CONFIG = {
            'log_level': 'INFO',
            'log_file': 'cricket_prediction.log',
            'max_log_size': 10485760,  # 10MB
            'backup_count': 5,
            'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
        
        # Database Configuration (for production)
        self.DB_CONFIG = {
            'database_url': os.getenv('DATABASE_URL', 'sqlite:///cricket_data.db'),
            'connection_pool_size': 10,
            'max_overflow': 20,
            'pool_timeout': 30
        }
        
        # Model Performance Thresholds
        self.PERFORMANCE_THRESHOLDS = {
            'minimum_accuracy': 0.75,
            'minimum_precision': 0.70,
            'minimum_recall': 0.70,
            'minimum_f1_score': 0.70,
            'minimum_roc_auc': 0.75,
            'retrain_threshold': 0.05  # Retrain if performance drops by 5%
        }
        
        # Prediction Configuration
        self.PREDICTION_CONFIG = {
            'confidence_threshold': 0.5,
            'uncertainty_threshold': 0.1,
            'ensemble_voting': 'weighted',
            'calibration_method': 'isotonic',
            'prediction_intervals': True,
            'confidence_level': 0.95
        }
        
        # Data Sources
        self.DATA_SOURCES = {
            'primary_api': 'cricapi',
            'backup_api': 'cricbuzz',
            'weather_api': 'openweathermap',
            'pitch_data': 'cricinfo',
            'historical_data': 'local_database'
        }
        
        # Feature Importance Configuration
        self.FEATURE_IMPORTANCE_CONFIG = {
            'method': 'permutation',  # 'permutation', 'shap', 'lime'
            'n_repeats': 10,
            'random_state': 42,
            'top_features': 20
        }
        
        # Cross-validation Configuration
        self.CV_CONFIG = {
            'n_splits': 5,
            'shuffle': True,
            'random_state': 42,
            'stratify': True
        }
        
        # Hyperparameter Tuning Configuration
        self.HYPERPARAMETER_CONFIG = {
            'search_method': 'random',  # 'grid', 'random', 'bayesian'
            'n_iter': 100,
            'cv_folds': 5,
            'scoring': 'roc_auc',
            'n_jobs': -1,
            'random_state': 42
        }
        
        # Real-time Prediction Configuration
        self.REALTIME_CONFIG = {
            'update_interval': 300,  # 5 minutes
            'match_tracking': True,
            'live_score_updates': True,
            'weather_updates': True,
            'auto_retrain': False,
            'retrain_schedule': '0 2 * * *'  # Daily at 2 AM
        }
        
        # Visualization Configuration
        self.VIZ_CONFIG = {
            'default_theme': 'plotly',
            'color_palette': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
            'figure_size': (12, 8),
            'dpi': 300,
            'save_format': 'png'
        }
        
        # Security Configuration
        self.SECURITY_CONFIG = {
            'api_rate_limit': 100,  # requests per minute
            'max_request_size': 1048576,  # 1MB
            'allowed_origins': ['*'],
            'secure_headers': True,
            'csrf_protection': True
        }
        
        # Deployment Configuration
        self.DEPLOYMENT_CONFIG = {
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'debug_mode': os.getenv('DEBUG', 'False').lower() == 'true',
            'host': '0.0.0.0',
            'port': 5000,
            'workers': 4,
            'max_workers': 8
        }
        
        # Model Versioning
        self.MODEL_VERSIONING = {
            'enable_versioning': True,
            'version_format': 'v{major}.{minor}.{patch}',
            'auto_increment': True,
            'keep_versions': 5,
            'backup_models': True
        }
        
        # Monitoring Configuration
        self.MONITORING_CONFIG = {
            'enable_monitoring': True,
            'metrics_endpoint': '/metrics',
            'health_check_endpoint': '/health',
            'performance_tracking': True,
            'error_tracking': True,
            'usage_analytics': True
        }
        
        # Notification Configuration
        self.NOTIFICATION_CONFIG = {
            'enable_notifications': False,
            'email_notifications': False,
            'slack_webhook': os.getenv('SLACK_WEBHOOK', ''),
            'alert_threshold': 0.05,
            'notification_frequency': 'daily'
        }
        
        # Backup Configuration
        self.BACKUP_CONFIG = {
            'enable_backup': True,
            'backup_frequency': 'daily',
            'backup_retention': 30,  # days
            'backup_location': 'backups/',
            'compress_backups': True
        }
        
        # Player Categories for Analysis
        self.PLAYER_CATEGORIES = {
            'formats': ['Test', 'ODI', 'T20', 'T10'],
            'roles': ['Batsman', 'Bowler', 'All-rounder', 'Wicket-keeper'],
            'experience_levels': ['Rookie', 'Experienced', 'Veteran'],
            'form_categories': ['Excellent', 'Good', 'Average', 'Poor']
        }
        
        # Venue Categories
        self.VENUE_CATEGORIES = {
            'pitch_types': ['Flat', 'Green', 'Dusty', 'Cracked'],
            'weather_conditions': ['Clear', 'Overcast', 'Light Rain', 'Heavy Rain'],
            'session_types': ['Morning', 'Afternoon', 'Evening', 'Night'],
            'venue_sizes': ['Small', 'Medium', 'Large']
        }
        
        # Model Ensemble Weights (auto-calculated if None)
        self.ENSEMBLE_WEIGHTS = {
            'tensorflow': None,
            'pytorch': None,
            'random_forest': None,
            'logistic_regression': None,
            'svm': None
        }
        
        # Alert Thresholds
        self.ALERT_THRESHOLDS = {
            'low_accuracy': 0.70,
            'high_error_rate': 0.30,
            'slow_prediction_time': 5.0,  # seconds
            'high_memory_usage': 0.80,  # 80% of available memory
            'disk_space_low': 0.90  # 90% disk usage
        }
        
        # Default Prediction Values
        self.DEFAULT_VALUES = {
            'century_probability': 0.1,
            'confidence_score': 0.5,
            'form_score': 0.5,
            'venue_advantage': 0.0,
            'weather_impact': 0.0,
            'pitch_impact': 0.0
        }
    
    def get_model_config(self, model_type):
        """Get configuration for specific model type"""
        return self.MODEL_CONFIG.get(model_type, {})
    
    def get_api_endpoint(self, endpoint_name, **kwargs):
        """Get API endpoint with formatting"""
        endpoint = self.API_ENDPOINTS.get(endpoint_name, '')
        if kwargs:
            endpoint = endpoint.format(**kwargs)
        return self.API_ENDPOINTS['base_url'] + endpoint
    
    def is_production(self):
        """Check if running in production environment"""
        return self.DEPLOYMENT_CONFIG['environment'] == 'production'
    
    def get_cache_key(self, prefix, identifier):
        """Generate cache key"""
        return f"{prefix}:{identifier}:{datetime.now().strftime('%Y%m%d%H')}"
    
    def get_model_version(self, major=1, minor=0, patch=0):
        """Get formatted model version"""
        return self.MODEL_VERSIONING['version_format'].format(
            major=major, minor=minor, patch=patch
        )
    
    def validate_config(self):
        """Validate configuration settings"""
        errors = []
        
        # Validate required API keys
        if self.API_KEY == 'demo_key':
            errors.append("Cricket API key not configured")
        
        # Validate model parameters
        for model_type, config in self.MODEL_CONFIG.items():
            if config.get('epochs', 0) <= 0:
                errors.append(f"Invalid epochs for {model_type}")
            if config.get('batch_size', 0) <= 0:
                errors.append(f"Invalid batch size for {model_type}")
        
        # Validate data split ratios
        if not (0 < self.DATA_CONFIG['train_test_split'] < 1):
            errors.append("Invalid train-test split ratio")
        
        # Validate thresholds
        for threshold_name, threshold_value in self.PERFORMANCE_THRESHOLDS.items():
            if not (0 <= threshold_value <= 1):
                errors.append(f"Invalid threshold: {threshold_name}")
        
        return errors
    
    def update_config(self, config_dict):
        """Update configuration with new values"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def export_config(self):
        """Export configuration as dictionary"""
        config_dict = {}
        for attr in dir(self):
            if not attr.startswith('_') and not callable(getattr(self, attr)):
                config_dict[attr] = getattr(self, attr)
        return config_dict
    
    def load_config_from_file(self, config_file):
        """Load configuration from JSON file"""
        import json
        try:
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            self.update_config(config_dict)
            return True
        except Exception as e:
            print(f"Error loading config from file: {e}")
            return False
    
    def save_config_to_file(self, config_file):
        """Save configuration to JSON file"""
        import json
        try:
            with open(config_file, 'w') as f:
                json.dump(self.export_config(), f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error saving config to file: {e}")
            return False