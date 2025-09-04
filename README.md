

# Cricket Century Prediction 🏏📊

# 🚀 Live demo
   
  view live demo https://reaishma.github.io/Cricket-century-prediction-/

## 🚀Overview

This is a comprehensive cricket century prediction platform built with Streamlit that uses machine learning to predict when cricket players are likely to score centuries. The system combines multiple model architectures including TensorFlow, PyTorch, scikit -learn and ensemble methods to provide accurate predictions based on historical player data and match conditions.

![overview](https://github.com/Reaishma/Cricket-century-prediction-/blob/main/Screenshot_20250904-144333_1.jpg)

## 🛠️System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application
- **Layout**: Wide layout with expandable sidebar for navigation
- **Components**: Interactive dashboards, real-time predictions, data visualization
- **Caching**: Streamlit resource caching for component initialization

### Backend Architecture
- **Model Layer**: Multiple ML models (TensorFlow, PyTorch, Ensemble)
- **Data Layer**: API client for cricket data, data processing pipeline
- **Feature Engineering**: Rolling averages, form features, player statistics
- **Utilities**: Visualization tools, model evaluation utilities

### Data Processing Pipeline
- **Data Sources**: Cricket API integration with fallback mock data
- **Processing**: Automated data cleaning, feature engineering, scaling
- **Storage**: In-memory processing with model persistence via joblib

## ✅Features Overview

### 🏏 Dashboard
- Live metrics showing active players, model accuracy, and predictions
- Performance charts tracking model accuracy over time
- Format-wise century distribution analysis
- Recent predictions table with confidence levels

### 🎯 Real-time Predictions

![Real time player](https://github.com/Reaishma/Cricket-century-prediction-/blob/main/Screenshot_20250904-144403_1.jpg)

- Interactive form to input player details and match conditions
- ML-powered prediction algorithm considering:
  - Player statistics (average, centuries, strike rate)
  - Match format (Test, ODI, T20, T10)
  - Weather conditions
  - Pitch type
  - Temperature
- Visual confidence meter and key factors analysis

### 👤 Player Analysis
- Search functionality for all players
- Detailed player profiles with comprehensive statistics
- Form trend analysis (last 10 matches)
- Venue-specific performance charts

### 🤖 Model Comparison

![model comparison](https://github.com/Reaishma/Cricket-century-prediction-/blob/main/Screenshot_20250904-144419_1.jpg)

- Performance metrics for TensorFlow, PyTorch, and Ensemble models
- Interactive comparison charts
- Accuracy, Precision, Recall, and F1 Score visualization

### 📈 Data Explorer
- Filterable player statistics table
- Country-wise century distribution
- Format-wise average scores
- Dynamic chart updates based on filters



## 📝Customization

### Adding New Players
Edit the `playersData` array in `script.js`:

```javascript
{
    id: "player_id",
    name: "Player Name",
    country: "Country",
    role: "Batsman/Bowler/All-rounder",
    matches: 100,
    runs: 5000,
    average: 50.0,
    centuries: 15,
    strikeRate: 85.5
}
```

### Modifying Prediction Algorithm
The prediction logic is in the `makePrediction()` function in `script.js`. You can adjust:
- Base prediction weights
- Format factors
- Weather impact
- Pitch condition effects
- Temperature influences

## ✅Key Components

### Machine Learning Models
1. **TensorFlow Model**: Deep neural network with batch normalization and dropout
2. **PyTorch Model**: Custom neural network with advanced training features
3. **Ensemble Model**: Combines multiple models using voting mechanisms
4. **Traditional ML**: Random Forest, Logistic Regression, SVM for ensemble

### Data Management
- **API Client**: Handles cricket data fetching with error handling and mock data fallback
- **Data Processor**: Manages data loading, cleaning, and preprocessing
- **Feature Engineer**: Creates rolling statistics, form features, and player metrics

### Visualization & Analysis
- **Visualizer**: Plotly-based interactive charts and graphs
- **Model Utils**: Comprehensive model evaluation and comparison tools
- **Real-time Dashboards**: Live prediction interfaces

## Data Flow

1. **Data Ingestion**: Cricket API client fetches player and match data
2. **Data Processing**: Raw data is cleaned and preprocessed
3. **Feature Engineering**: Statistical features and rolling averages are created
4. **Model Training**: Multiple models are trained on processed features
5. **Prediction**: Ensemble model combines predictions for final output
6. **Visualization**: Results are displayed through interactive Streamlit interface

## External Dependencies

### APIs
- **Cricket API**: Primary data source for player statistics and match data
- **Weather API**: Additional context for match conditions
- **Fallback System**: Mock data generation when APIs are unavailable

### Machine Learning Libraries
- **TensorFlow/Keras**: Deep learning model implementation
- **PyTorch**: Alternative deep learning framework
- **Scikit-learn**: Traditional ML algorithms and utilities
- **Plotly**: Interactive visualization library

### Data Processing
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Streamlit**: Web application framework
- **Joblib**: Model persistence and serialization

## Deployment Strategy

### Configuration Management
- **Environment Variables**: API keys and sensitive configuration
- **Settings Class**: Centralized configuration management
- **Model Parameters**: Configurable hyperparameters for all models

### Model Management
- **Multi-Model Architecture**: Parallel training of different model types
- **Performance Monitoring**: Comprehensive evaluation metrics
- **Model Persistence**: Automatic saving and loading of trained models

### Scalability Considerations
- **Caching Strategy**: Streamlit resource caching for expensive operations
- **Batch Processing**: Efficient data processing for large datasets
- **Error Handling**: Robust error handling with graceful degradation

### Development Features
- **Mock Data**: Comprehensive fallback system for development
- **Modular Design**: Clear separation of concerns across components
- **Configuration Flexibility**: Easy adjustment of model parameters and thresholds

## 🤝 Contributions 

Development Process
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request


## 📍Example Output
[!https://drive.google.com/uc?id=1R2GfJowvzQ1pcx4DYnOewfAwy_9KjwZS](https://drive.google.com/file/d/1R2GfJowvzQ1pcx4DYnOewfAwy_9KjwZS/view) 🎥

## Example Use Cases
- Predicting centuries in international cricket matches 🏏
- Analyzing team performance and strategy 📈
- Identifying key player statistics that influence century predictions 🔍


## Author
  *Reaishma N* 🙋‍♀️ [GitHub](https://github.com/Reaishma)

## 📄License
  This project is licensed under MIT Licence see the [licence](https://github.com/Reaishma/Cricket-century-prediction-/blob/main/LICENSE) file for details 

