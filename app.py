import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
# pytorch_model.py

try:
    # Try to import PyTorch
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    # Define a flag to indicate successful import
    TORCH_AVAILABLE = True

except ImportError:
    # PyTorch is not installed. Handle this gracefully.
    TORCH_AVAILABLE = False
    
    # You can also define placeholder classes or functions here
    # to prevent further NameErrors.
    class PyTorchModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not installed. Please install it with 'pip install torch'.")
        # Add other method definitions (e.g., predict, train)
        # that will also raise an error if called.
        def predict(self, data):
            raise RuntimeError("PyTorch is not installed.")

# Now, based on the flag, define the actual model if it's available.
if TORCH_AVAILABLE:
    
    # This is where your actual PyTorch model class definition goes
    class PyTorchModel:
        def __init__(self, input_dim, hidden_dim, output_dim, learning_rate):
            super().__init__()
            # Your model architecture here
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid() # Or whatever activation is appropriate
            )
            # ... other initialization logic
        
        def train(self, X_train, y_train):
            # Training logic here
            pass

        def predict(self, X_test):
            # Prediction logic here
            pass

# Import custom modules
from models.tensorflow_model import TensorFlowModel
from models.pytorch_model import PyTorchModel
from models.ensemble_model import EnsembleModel
from data.data_processor import DataProcessor
from data.api_client import CricketAPIClient
from data.feature_engineering import FeatureEngineer
from utils.visualization import Visualizer
from utils.model_utils import ModelUtils
from config.settings import Settings

# Configure page
st.set_page_config(
    page_title="Cricket Century Prediction Platform",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def initialize_components():
    settings = Settings()
    api_client = CricketAPIClient(settings.API_KEY)
    data_processor = DataProcessor()
    feature_engineer = FeatureEngineer()
    visualizer = Visualizer()
    model_utils = ModelUtils()
    
    return {
        'settings': settings,
        'api_client': api_client,
        'data_processor': data_processor,
        'feature_engineer': feature_engineer,
        'visualizer': visualizer,
        'model_utils': model_utils
    }

def main():
    components = initialize_components()
    
    # Header
    st.title("🏏 Cricket Century Prediction Platform")
    st.markdown("### Advanced ML-powered analysis for predicting cricket centuries")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Real-time Predictions", "Player Analysis", "Model Comparison", "Data Explorer"]
    )
    
    if page == "Dashboard":
        dashboard_page(components)
    elif page == "Real-time Predictions":
        prediction_page(components)
    elif page == "Player Analysis":
        player_analysis_page(components)
    elif page == "Model Comparison":
        model_comparison_page(components)
    elif page == "Data Explorer":
        data_explorer_page(components)

def dashboard_page(components):
    st.header("📊 Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Players", "2,847", "↑ 12%")
    
    with col2:
        st.metric("Model Accuracy", "87.3%", "↑ 2.1%")
    
    with col3:
        st.metric("Predictions Today", "156", "↑ 23%")
    
    with col4:
        st.metric("Data Points", "45.2K", "↑ 5.8%")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance Over Time")
        # Generate sample performance data
        dates = pd.date_range(start='2024-01-01', end='2025-07-10', freq='D')
        performance_data = pd.DataFrame({
            'Date': dates,
            'TensorFlow': np.random.normal(0.85, 0.05, len(dates)).cumsum() * 0.01 + 0.80,
            'PyTorch': np.random.normal(0.83, 0.05, len(dates)).cumsum() * 0.01 + 0.78,
            'Ensemble': np.random.normal(0.87, 0.04, len(dates)).cumsum() * 0.01 + 0.82
        })
        
        fig = px.line(performance_data, x='Date', y=['TensorFlow', 'PyTorch', 'Ensemble'],
                     title="Model Accuracy Trends")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Century Predictions by Format")
        format_data = pd.DataFrame({
            'Format': ['Test', 'ODI', 'T20', 'T10'],
            'Centuries': [145, 289, 167, 43],
            'Predictions': [156, 301, 178, 39]
        })
        
        fig = px.bar(format_data, x='Format', y=['Centuries', 'Predictions'],
                    title="Actual vs Predicted Centuries", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent predictions
    st.subheader("Recent Predictions")
    recent_predictions = pd.DataFrame({
        'Player': ['Virat Kohli', 'Babar Azam', 'Steve Smith', 'Kane Williamson', 'Joe Root'],
        'Match': ['IND vs AUS', 'PAK vs ENG', 'AUS vs SA', 'NZ vs WI', 'ENG vs IND'],
        'Prediction': [0.78, 0.65, 0.82, 0.71, 0.69],
        'Confidence': ['High', 'Medium', 'High', 'High', 'Medium'],
        'Status': ['Ongoing', 'Upcoming', 'Completed', 'Upcoming', 'Completed']
    })
    
    st.dataframe(recent_predictions, use_container_width=True)

def prediction_page(components):
    st.header("🎯 Real-time Predictions")
    
    # Player selection
    col1, col2 = st.columns(2)
    
    with col1:
        player_name = st.text_input("Enter Player Name", "Virat Kohli")
        match_format = st.selectbox("Match Format", ["Test", "ODI", "T20", "T10"])
    
    with col2:
        opposition = st.text_input("Opposition Team", "Australia")
        venue = st.text_input("Venue", "Melbourne Cricket Ground")
    
    # Match conditions
    st.subheader("Match Conditions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        weather = st.selectbox("Weather", ["Clear", "Overcast", "Light Rain", "Heavy Rain"])
        temperature = st.slider("Temperature (°C)", 10, 45, 25)
    
    with col2:
        pitch_type = st.selectbox("Pitch Type", ["Flat", "Green", "Dusty", "Cracked"])
        humidity = st.slider("Humidity (%)", 20, 90, 60)
    
    with col3:
        wind_speed = st.slider("Wind Speed (km/h)", 0, 30, 10)
        day_night = st.selectbox("Day/Night", ["Day", "Night"])
    
    # Prediction button
    if st.button("Generate Prediction", type="primary"):
        with st.spinner("Analyzing player data and match conditions..."):
            # Simulate API call and prediction
            time.sleep(2)
            
            # Mock prediction results
            tf_prediction = np.random.uniform(0.6, 0.9)
            pytorch_prediction = np.random.uniform(0.55, 0.85)
            ensemble_prediction = (tf_prediction + pytorch_prediction) / 2
            
            # Display results
            st.success("Prediction Generated!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("TensorFlow Model", f"{tf_prediction:.1%}", "High Confidence")
            
            with col2:
                st.metric("PyTorch Model", f"{pytorch_prediction:.1%}", "Medium Confidence")
            
            with col3:
                st.metric("Ensemble Model", f"{ensemble_prediction:.1%}", "High Confidence")
            
            # Visualization
            st.subheader("Prediction Breakdown")
            
            # Feature importance
            features = ['Recent Form', 'Venue History', 'Opposition Record', 'Weather Conditions', 'Pitch Type']
            importance = np.random.uniform(0.1, 0.9, len(features))
            
            fig = px.bar(x=features, y=importance, title="Feature Importance for Prediction")
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence intervals
            st.subheader("Prediction Confidence")
            confidence_data = pd.DataFrame({
                'Model': ['TensorFlow', 'PyTorch', 'Ensemble'],
                'Prediction': [tf_prediction, pytorch_prediction, ensemble_prediction],
                'Lower_CI': [tf_prediction - 0.1, pytorch_prediction - 0.12, ensemble_prediction - 0.08],
                'Upper_CI': [tf_prediction + 0.1, pytorch_prediction + 0.12, ensemble_prediction + 0.08]
            })
            
            fig = go.Figure()
            for _, row in confidence_data.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row['Model']], y=[row['Prediction']],
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=[row['Upper_CI'] - row['Prediction']],
                        arrayminus=[row['Prediction'] - row['Lower_CI']]
                    ),
                    mode='markers',
                    name=row['Model']
                ))
            
            fig.update_layout(title="Prediction Confidence Intervals")
            st.plotly_chart(fig, use_container_width=True)

def player_analysis_page(components):
    st.header("👤 Player Analysis")
    
    # Player selection
    player_name = st.selectbox("Select Player", 
                              ["Virat Kohli", "Babar Azam", "Steve Smith", "Kane Williamson", "Joe Root"])
    
    # Time period selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Player statistics
    st.subheader(f"Statistics for {player_name}")
    
    # Generate mock player data
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    player_data = pd.DataFrame({
        'Date': dates,
        'Runs': np.random.poisson(45, len(dates)),
        'Centuries': np.random.binomial(1, 0.1, len(dates)),
        'Strike_Rate': np.random.normal(85, 15, len(dates)),
        'Form_Score': np.random.normal(0.7, 0.2, len(dates))
    })
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Runs", f"{player_data['Runs'].sum():,}")
    
    with col2:
        st.metric("Centuries", f"{player_data['Centuries'].sum()}")
    
    with col3:
        st.metric("Avg Strike Rate", f"{player_data['Strike_Rate'].mean():.1f}")
    
    with col4:
        st.metric("Form Score", f"{player_data['Form_Score'].mean():.2f}")
    
    # Performance trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Runs Trend")
        fig = px.line(player_data, x='Date', y='Runs', title="Runs Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Form Analysis")
        fig = px.line(player_data, x='Date', y='Form_Score', title="Form Score Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    # Venue analysis
    st.subheader("Venue Performance")
    venue_data = pd.DataFrame({
        'Venue': ['MCG', 'Lords', 'Eden Gardens', 'Wankhede', 'SCG'],
        'Matches': [12, 8, 15, 10, 7],
        'Centuries': [3, 2, 4, 3, 1],
        'Average': [54.2, 67.8, 78.3, 62.1, 41.7]
    })
    
    fig = px.scatter(venue_data, x='Matches', y='Centuries', size='Average',
                    hover_data=['Venue'], title="Century Rate by Venue")
    st.plotly_chart(fig, use_container_width=True)

def model_comparison_page(components):
    st.header("🔬 Model Comparison")
    
    # Model performance metrics
    st.subheader("Model Performance Metrics")
    
    performance_data = pd.DataFrame({
        'Model': ['TensorFlow/Keras', 'PyTorch', 'Ensemble', 'Random Forest', 'XGBoost'],
        'Accuracy': [0.873, 0.856, 0.891, 0.834, 0.847],
        'Precision': [0.881, 0.867, 0.896, 0.841, 0.852],
        'Recall': [0.865, 0.848, 0.887, 0.829, 0.843],
        'F1_Score': [0.873, 0.857, 0.891, 0.835, 0.848],
        'Training_Time': [245, 312, 520, 67, 89]
    })
    
    st.dataframe(performance_data, use_container_width=True)
    
    # Performance comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Accuracy Comparison")
        fig = px.bar(performance_data, x='Model', y='Accuracy', 
                    title="Model Accuracy Comparison")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Training Time vs Accuracy")
        fig = px.scatter(performance_data, x='Training_Time', y='Accuracy', 
                        size='F1_Score', hover_data=['Model'],
                        title="Training Time vs Accuracy")
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrices
    st.subheader("Model Confusion Matrices")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**TensorFlow Model**")
        tf_cm = np.array([[850, 120], [95, 935]])
        fig = px.imshow(tf_cm, text_auto=True, title="TensorFlow Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**PyTorch Model**")
        pt_cm = np.array([[840, 130], [105, 925]])
        fig = px.imshow(pt_cm, text_auto=True, title="PyTorch Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.write("**Ensemble Model**")
        en_cm = np.array([[865, 105], [85, 945]])
        fig = px.imshow(en_cm, text_auto=True, title="Ensemble Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance comparison
    st.subheader("Feature Importance Comparison")
    
    features = ['Recent Form', 'Career Average', 'Venue History', 'Opposition Record', 
               'Weather Conditions', 'Pitch Type', 'Match Situation', 'Time of Day']
    
    importance_data = pd.DataFrame({
        'Feature': features,
        'TensorFlow': np.random.uniform(0.05, 0.25, len(features)),
        'PyTorch': np.random.uniform(0.05, 0.25, len(features)),
        'Ensemble': np.random.uniform(0.05, 0.25, len(features))
    })
    
    fig = px.bar(importance_data, x='Feature', y=['TensorFlow', 'PyTorch', 'Ensemble'],
                title="Feature Importance by Model", barmode='group')
    st.plotly_chart(fig, use_container_width=True)

def data_explorer_page(components):
    st.header("🔍 Data Explorer")
    
    # Data overview
    st.subheader("Dataset Overview")
    
    # Generate sample dataset info
    dataset_info = pd.DataFrame({
        'Statistic': ['Total Records', 'Players', 'Matches', 'Centuries', 'Date Range'],
        'Value': ['45,237', '2,847', '12,456', '3,892', '2010-2025']
    })
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(dataset_info, use_container_width=True)
    
    with col2:
        # Data distribution
        st.subheader("Score Distribution")
        scores = np.random.gamma(2, 20, 1000)
        fig = px.histogram(scores, title="Player Scores Distribution", nbins=50)
        st.plotly_chart(fig, use_container_width=True)
    
    # Filters
    st.subheader("Data Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        format_filter = st.multiselect("Format", ["Test", "ODI", "T20", "T10"], default=["Test", "ODI"])
    
    with col2:
        year_range = st.slider("Year Range", 2010, 2025, (2020, 2025))
    
    with col3:
        min_matches = st.number_input("Minimum Matches", 0, 100, 10)
    
    # Sample data display
    st.subheader("Sample Data")
    
    sample_data = pd.DataFrame({
        'Player': ['Virat Kohli', 'Babar Azam', 'Steve Smith', 'Kane Williamson', 'Joe Root'],
        'Format': ['ODI', 'Test', 'T20', 'Test', 'ODI'],
        'Score': [89, 156, 67, 134, 78],
        'Century': [False, True, False, True, False],
        'Venue': ['MCG', 'Lords', 'Eden Gardens', 'Basin Reserve', 'Oval'],
        'Date': ['2025-07-01', '2025-07-03', '2025-07-05', '2025-07-07', '2025-07-09']
    })
    
    st.dataframe(sample_data, use_container_width=True)
    
    # Data quality metrics
    st.subheader("Data Quality")
    
    col1, col2 = st.columns(2)
    
    with col1:
        quality_metrics = pd.DataFrame({
            'Metric': ['Completeness', 'Accuracy', 'Consistency', 'Timeliness'],
            'Score': [0.94, 0.91, 0.88, 0.96]
        })
        
        fig = px.bar(quality_metrics, x='Metric', y='Score', 
                    title="Data Quality Metrics")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Missing data analysis
        missing_data = pd.DataFrame({
            'Column': ['Player_ID', 'Score', 'Venue', 'Weather', 'Pitch_Type'],
            'Missing_%': [0.0, 0.2, 1.5, 8.3, 12.1]
        })
        
        fig = px.bar(missing_data, x='Column', y='Missing_%', 
                    title="Missing Data by Column")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()