import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st

class Visualizer:
    def __init__(self):
        self.colors = {
            'primary': '#FF6B6B',
            'secondary': '#4ECDC4',
            'accent': '#45B7D1',
            'success': '#96CEB4',
            'warning': '#FFEAA7',
            'danger': '#DDA0DD'
        }
    
    def plot_player_form_trend(self, player_data, player_name):
        """Plot player's form trend over time"""
        fig = go.Figure()
        
        # Add runs trend
        fig.add_trace(go.Scatter(
            x=player_data['date'],
            y=player_data['runs'],
            mode='lines+markers',
            name='Runs',
            line=dict(color=self.colors['primary'], width=2),
            marker=dict(size=6, color=self.colors['primary'])
        ))
        
        # Add century markers
        centuries = player_data[player_data['century'] == 1]
        if not centuries.empty:
            fig.add_trace(go.Scatter(
                x=centuries['date'],
                y=centuries['runs'],
                mode='markers',
                name='Centuries',
                marker=dict(
                    size=12,
                    color=self.colors['success'],
                    symbol='star',
                    line=dict(width=2, color='white')
                )
            ))
        
        # Add moving average
        if len(player_data) > 5:
            player_data['moving_avg'] = player_data['runs'].rolling(window=5).mean()
            fig.add_trace(go.Scatter(
                x=player_data['date'],
                y=player_data['moving_avg'],
                mode='lines',
                name='5-match Moving Average',
                line=dict(color=self.colors['secondary'], width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=f'{player_name} - Performance Trend',
            xaxis_title='Date',
            yaxis_title='Runs',
            hovermode='x unified',
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def plot_venue_performance(self, venue_data, player_name):
        """Plot player's performance at different venues"""
        fig = go.Figure()
        
        # Bar chart of average runs by venue
        fig.add_trace(go.Bar(
            x=venue_data['venue'],
            y=venue_data['average_runs'],
            name='Average Runs',
            marker_color=self.colors['primary'],
            text=venue_data['average_runs'].round(1),
            textposition='auto'
        ))
        
        # Add century rate as secondary y-axis
        fig.add_trace(go.Scatter(
            x=venue_data['venue'],
            y=venue_data['century_rate'] * 100,
            mode='lines+markers',
            name='Century Rate (%)',
            yaxis='y2',
            line=dict(color=self.colors['accent'], width=3),
            marker=dict(size=8, color=self.colors['accent'])
        ))
        
        fig.update_layout(
            title=f'{player_name} - Venue Performance',
            xaxis_title='Venue',
            yaxis=dict(title='Average Runs', side='left'),
            yaxis2=dict(title='Century Rate (%)', side='right', overlaying='y'),
            hovermode='x unified',
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def plot_opposition_analysis(self, opposition_data, player_name):
        """Plot player's performance against different oppositions"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Runs', 'Century Rate', 'Strike Rate', 'Matches Played'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Average runs
        fig.add_trace(
            go.Bar(x=opposition_data['opposition'], y=opposition_data['avg_runs'],
                   name='Avg Runs', marker_color=self.colors['primary']),
            row=1, col=1
        )
        
        # Century rate
        fig.add_trace(
            go.Bar(x=opposition_data['opposition'], y=opposition_data['century_rate'],
                   name='Century Rate', marker_color=self.colors['success']),
            row=1, col=2
        )
        
        # Strike rate
        fig.add_trace(
            go.Bar(x=opposition_data['opposition'], y=opposition_data['strike_rate'],
                   name='Strike Rate', marker_color=self.colors['accent']),
            row=2, col=1
        )
        
        # Matches played
        fig.add_trace(
            go.Bar(x=opposition_data['opposition'], y=opposition_data['matches'],
                   name='Matches', marker_color=self.colors['warning']),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'{player_name} - Opposition Analysis',
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def plot_weather_impact(self, weather_data):
        """Plot the impact of weather conditions on performance"""
        fig = go.Figure()
        
        # Box plot for runs by weather condition
        for condition in weather_data['weather'].unique():
            condition_data = weather_data[weather_data['weather'] == condition]
            fig.add_trace(go.Box(
                y=condition_data['runs'],
                name=condition,
                boxpoints='outliers',
                marker_color=self.colors['primary']
            ))
        
        fig.update_layout(
            title='Runs Distribution by Weather Condition',
            xaxis_title='Weather Condition',
            yaxis_title='Runs',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def plot_pitch_analysis(self, pitch_data):
        """Plot performance analysis by pitch type"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Average Score by Pitch Type', 'Century Rate by Pitch Type'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Average score
        fig.add_trace(
            go.Bar(x=pitch_data['pitch_type'], y=pitch_data['avg_score'],
                   name='Avg Score', marker_color=self.colors['primary']),
            row=1, col=1
        )
        
        # Century rate
        fig.add_trace(
            go.Bar(x=pitch_data['pitch_type'], y=pitch_data['century_rate'],
                   name='Century Rate', marker_color=self.colors['success']),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Performance Analysis by Pitch Type',
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def plot_model_comparison(self, model_results):
        """Plot comparison of different models"""
        models = list(model_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [model_results[model][metric] for model in models]
            fig.add_trace(go.Bar(
                x=models,
                y=values,
                name=metric.title(),
                text=[f'{v:.3f}' for v in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def plot_feature_importance(self, feature_importance_df, top_n=15):
        """Plot feature importance"""
        # Sort by importance and take top N
        top_features = feature_importance_df.head(top_n)
        
        fig = go.Figure(go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker_color=self.colors['primary'],
            text=top_features['importance'].round(3),
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def plot_prediction_confidence(self, predictions, confidence_intervals):
        """Plot predictions with confidence intervals"""
        fig = go.Figure()
        
        x_values = list(range(len(predictions)))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=x_values + x_values[::-1],
            y=confidence_intervals['upper'] + confidence_intervals['lower'][::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            showlegend=True
        ))
        
        # Add predictions
        fig.add_trace(go.Scatter(
            x=x_values,
            y=predictions,
            mode='lines+markers',
            name='Predictions',
            line=dict(color=self.colors['primary'], width=2),
            marker=dict(size=6, color=self.colors['primary'])
        ))
        
        fig.update_layout(
            title='Century Predictions with Confidence Intervals',
            xaxis_title='Sample Index',
            yaxis_title='Century Probability',
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def plot_correlation_matrix(self, df, features):
        """Plot correlation matrix of features"""
        correlation_matrix = df[features].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            showscale=True
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def plot_player_radar(self, player_stats, player_name):
        """Create radar chart for player statistics"""
        categories = list(player_stats.keys())
        values = list(player_stats.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=player_name,
            line=dict(color=self.colors['primary']),
            fillcolor=f'rgba({int(self.colors["primary"][1:3], 16)}, {int(self.colors["primary"][3:5], 16)}, {int(self.colors["primary"][5:7], 16)}, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=f'{player_name} - Performance Radar',
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def plot_time_series_prediction(self, historical_data, predictions, future_dates):
        """Plot time series with predictions"""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['runs'],
            mode='lines+markers',
            name='Historical Performance',
            line=dict(color=self.colors['primary'], width=2),
            marker=dict(size=4, color=self.colors['primary'])
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name='Predicted Performance',
            line=dict(color=self.colors['accent'], width=2, dash='dash'),
            marker=dict(size=6, color=self.colors['accent'])
        ))
        
        fig.update_layout(
            title='Performance Trend with Predictions',
            xaxis_title='Date',
            yaxis_title='Runs',
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def plot_confusion_matrix(self, cm, model_name):
        """Plot confusion matrix"""
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=['Not Century', 'Century'],
            y=['Not Century', 'Century'],
            annotation_text=cm,
            colorscale='Blues',
            showscale=True
        )
        
        fig.update_layout(
            title=f'{model_name} - Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def plot_roc_curve(self, fpr, tpr, auc_score, model_name):
        """Plot ROC curve"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {auc_score:.3f})',
            line=dict(color=self.colors['primary'], width=2)
        ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title=f'{model_name} - ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_dashboard_metrics(self, metrics_data):
        """Create metrics dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Trend', 'Prediction Distribution', 
                          'Feature Usage', 'Model Performance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Accuracy trend
        fig.add_trace(
            go.Scatter(x=metrics_data['dates'], y=metrics_data['accuracy'],
                      name='Accuracy', line=dict(color=self.colors['primary'])),
            row=1, col=1
        )
        
        # Prediction distribution
        fig.add_trace(
            go.Histogram(x=metrics_data['predictions'], name='Predictions',
                        marker_color=self.colors['secondary']),
            row=1, col=2
        )
        
        # Feature usage
        fig.add_trace(
            go.Bar(x=metrics_data['features'], y=metrics_data['usage'],
                   name='Usage', marker_color=self.colors['accent']),
            row=2, col=1
        )
        
        # Model performance
        fig.add_trace(
            go.Bar(x=metrics_data['models'], y=metrics_data['performance'],
                   name='Performance', marker_color=self.colors['success']),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Cricket Century Prediction Dashboard',
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
