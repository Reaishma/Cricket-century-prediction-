import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.scalers = {}
        
    def create_rolling_features(self, df, player_col='player', date_col='date', 
                               target_cols=['runs', 'strike_rate'], windows=[5, 10, 20]):
        """Create rolling average features for players"""
        df_sorted = df.sort_values([player_col, date_col])
        
        for window in windows:
            for col in target_cols:
                if col in df.columns:
                    df_sorted[f'{col}_rolling_{window}'] = (
                        df_sorted.groupby(player_col)[col]
                        .rolling(window=window, min_periods=1)
                        .mean()
                        .reset_index(level=0, drop=True)
                    )
        
        return df_sorted
    
    def create_form_features(self, df, player_col='player', date_col='date'):
        """Create player form features"""
        df_sorted = df.sort_values([player_col, date_col])
        
        # Recent form score (weighted average of last 10 matches)
        def calculate_form_score(group):
            if len(group) < 3:
                return [0.5] * len(group)  # Default neutral form
            
            form_scores = []
            for i in range(len(group)):
                if i < 2:
                    form_scores.append(0.5)
                else:
                    # Calculate weighted average of last few matches
                    recent_scores = group.iloc[max(0, i-9):i]['runs'].values
                    weights = np.exp(np.linspace(-2, 0, len(recent_scores)))
                    weights /= weights.sum()
                    
                    weighted_avg = np.average(recent_scores, weights=weights)
                    # Normalize to 0-1 scale
                    form_score = min(1.0, max(0.0, weighted_avg / 100))
                    form_scores.append(form_score)
            
            return form_scores
        
        df_sorted['form_score'] = (
            df_sorted.groupby(player_col)
            .apply(lambda x: pd.Series(calculate_form_score(x), index=x.index))
            .reset_index(level=0, drop=True)
        )
        
        # Consistency score (inverse of coefficient of variation)
        def calculate_consistency(group):
            if len(group) < 5:
                return [0.5] * len(group)
            
            consistency_scores = []
            for i in range(len(group)):
                if i < 4:
                    consistency_scores.append(0.5)
                else:
                    recent_scores = group.iloc[max(0, i-9):i]['runs'].values
                    if len(recent_scores) > 1 and np.std(recent_scores) > 0:
                        cv = np.std(recent_scores) / np.mean(recent_scores)
                        consistency = 1 / (1 + cv)  # Higher consistency = lower CV
                    else:
                        consistency = 0.5
                    consistency_scores.append(consistency)
            
            return consistency_scores
        
        df_sorted['consistency_score'] = (
            df_sorted.groupby(player_col)
            .apply(lambda x: pd.Series(calculate_consistency(x), index=x.index))
            .reset_index(level=0, drop=True)
        )
        
        return df_sorted
    
    def create_venue_features(self, df, player_col='player', venue_col='venue'):
        """Create venue-specific features"""
        # Calculate player's historical performance at each venue
        venue_stats = df.groupby([player_col, venue_col]).agg({
            'runs': ['mean', 'std', 'count'],
            'century': 'mean'
        }).reset_index()
        
        venue_stats.columns = [
            player_col, venue_col, 'venue_avg_runs', 'venue_std_runs', 
            'venue_matches', 'venue_century_rate'
        ]
        
        # Merge back to original dataframe
        df_with_venue = df.merge(venue_stats, on=[player_col, venue_col], how='left')
        
        # Fill NaN values for new player-venue combinations
        df_with_venue['venue_avg_runs'] = df_with_venue['venue_avg_runs'].fillna(
            df_with_venue['career_average']
        )
        df_with_venue['venue_std_runs'] = df_with_venue['venue_std_runs'].fillna(20)
        df_with_venue['venue_matches'] = df_with_venue['venue_matches'].fillna(0)
        df_with_venue['venue_century_rate'] = df_with_venue['venue_century_rate'].fillna(0.1)
        
        return df_with_venue
    
    def create_opposition_features(self, df, player_col='player', opp_col='opposition'):
        """Create opposition-specific features"""
        # Calculate player's historical performance against each opposition
        opp_stats = df.groupby([player_col, opp_col]).agg({
            'runs': ['mean', 'std', 'count'],
            'century': 'mean'
        }).reset_index()
        
        opp_stats.columns = [
            player_col, opp_col, 'opp_avg_runs', 'opp_std_runs', 
            'opp_matches', 'opp_century_rate'
        ]
        
        # Merge back to original dataframe
        df_with_opp = df.merge(opp_stats, on=[player_col, opp_col], how='left')
        
        # Fill NaN values for new player-opposition combinations
        df_with_opp['opp_avg_runs'] = df_with_opp['opp_avg_runs'].fillna(
            df_with_opp['career_average']
        )
        df_with_opp['opp_std_runs'] = df_with_opp['opp_std_runs'].fillna(20)
        df_with_opp['opp_matches'] = df_with_opp['opp_matches'].fillna(0)
        df_with_opp['opp_century_rate'] = df_with_opp['opp_century_rate'].fillna(0.1)
        
        return df_with_opp
    
    def create_weather_features(self, df):
        """Create weather-related features"""
        # Temperature comfort index
        df['temp_comfort'] = 1 - abs(df['temperature'] - 25) / 25
        df['temp_comfort'] = df['temp_comfort'].clip(0, 1)
        
        # Humidity comfort index
        df['humidity_comfort'] = 1 - abs(df['humidity'] - 50) / 50
        df['humidity_comfort'] = df['humidity_comfort'].clip(0, 1)
        
        # Wind impact (higher wind = more difficult batting)
        df['wind_impact'] = df['wind_speed'] / 30
        df['wind_impact'] = df['wind_impact'].clip(0, 1)
        
        # Weather favorability for batting
        weather_scores = {
            'Clear': 1.0,
            'Overcast': 0.7,
            'Light Rain': 0.4,
            'Heavy Rain': 0.1
        }
        df['weather_batting_score'] = df['weather'].map(weather_scores)
        
        # Combined weather score
        df['weather_score'] = (
            df['temp_comfort'] * 0.3 +
            df['humidity_comfort'] * 0.2 +
            (1 - df['wind_impact']) * 0.2 +
            df['weather_batting_score'] * 0.3
        )
        
        return df
    
    def create_pitch_features(self, df):
        """Create pitch-related features"""
        # Pitch batting difficulty
        pitch_batting_scores = {
            'Flat': 1.0,
            'Dusty': 0.7,
            'Green': 0.5,
            'Cracked': 0.3
        }
        df['pitch_batting_score'] = df['pitch_type'].map(pitch_batting_scores)
        
        return df
    
    def create_match_context_features(self, df):
        """Create match context features"""
        # Batting position impact
        df['batting_pos_score'] = np.where(
            df['batting_position'] <= 3, 1.0,
            np.where(df['batting_position'] <= 5, 0.8, 0.6)
        )
        
        # Innings pressure
        df['innings_pressure'] = np.where(
            df['innings'] == 1, 0.5,  # Less pressure batting first
            np.where(df['target_score'] > 250, 0.8, 0.6)  # More pressure chasing high scores
        )
        
        # Day-night factor
        df['day_night_factor'] = np.where(
            df['is_day_night'] == 1, 0.9, 1.0  # Slightly harder in day-night matches
        )
        
        # Session factor
        session_scores = {
            'Morning': 0.8,
            'Afternoon': 1.0,
            'Evening': 0.7
        }
        df['session_score'] = df['session'].map(session_scores)
        
        return df
    
    def create_time_features(self, df, date_col='date'):
        """Create time-based features"""
        if date_col in df.columns:
            df['year'] = df[date_col].dt.year
            df['month'] = df[date_col].dt.month
            df['day_of_week'] = df[date_col].dt.dayofweek
            df['day_of_year'] = df[date_col].dt.dayofyear
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Seasonal effects
            df['season'] = df['month'].apply(self._get_season)
            
            # Career stage (early, mid, late)
            df['career_stage'] = pd.cut(
                df['career_matches'], 
                bins=[0, 50, 150, 1000], 
                labels=['Early', 'Mid', 'Late']
            )
        
        return df
    
    def _get_season(self, month):
        """Get season based on month"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
    
    def create_interaction_features(self, df):
        """Create interaction features"""
        # Player quality vs conditions
        df['quality_vs_conditions'] = (
            df['career_average'] / 50 * df['weather_score'] * df['pitch_batting_score']
        )
        
        # Form vs venue familiarity
        df['form_venue_interaction'] = df['form_score'] * df['venue_century_rate']
        
        # Experience vs pressure
        df['experience_pressure'] = (
            df['career_matches'] / 200 * (1 - df['innings_pressure'])
        )
        
        # Recent form vs opposition record
        df['form_vs_opposition'] = df['form_score'] * df['opp_century_rate']
        
        return df
    
    def create_momentum_features(self, df, player_col='player', date_col='date'):
        """Create momentum-based features"""
        df_sorted = df.sort_values([player_col, date_col])
        
        # Consecutive matches momentum
        def calculate_momentum(group):
            momentum_scores = []
            for i in range(len(group)):
                if i < 3:
                    momentum_scores.append(0.5)
                else:
                    # Look at last 3 matches
                    recent_scores = group.iloc[i-3:i]['runs'].values
                    recent_centuries = group.iloc[i-3:i]['century'].values
                    
                    # Calculate momentum based on trend
                    if len(recent_scores) >= 3:
                        # Simple trend calculation
                        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                        momentum = 0.5 + np.tanh(trend / 20) * 0.3
                        
                        # Boost for recent centuries
                        if recent_centuries.sum() > 0:
                            momentum += 0.2 * recent_centuries.sum() / len(recent_centuries)
                        
                        momentum = max(0, min(1, momentum))
                    else:
                        momentum = 0.5
                    
                    momentum_scores.append(momentum)
            
            return momentum_scores
        
        df_sorted['momentum_score'] = (
            df_sorted.groupby(player_col)
            .apply(lambda x: pd.Series(calculate_momentum(x), index=x.index))
            .reset_index(level=0, drop=True)
        )
        
        return df_sorted
    
    def create_advanced_features(self, df):
        """Create advanced engineered features"""
        # Apply all feature engineering steps
        df = self.create_rolling_features(df)
        df = self.create_form_features(df)
        df = self.create_venue_features(df)
        df = self.create_opposition_features(df)
        df = self.create_weather_features(df)
        df = self.create_pitch_features(df)
        df = self.create_match_context_features(df)
        df = self.create_time_features(df)
        df = self.create_interaction_features(df)
        df = self.create_momentum_features(df)
        
        return df
    
    def select_features(self, df, target_col='century'):
        """Select most relevant features"""
        # Define feature categories
        basic_features = [
            'career_average', 'career_centuries', 'career_matches',
            'recent_average', 'recent_centuries', 'batting_position'
        ]
        
        form_features = [
            'form_score', 'consistency_score', 'momentum_score'
        ]
        
        context_features = [
            'venue_avg_runs', 'venue_century_rate', 'opp_avg_runs', 'opp_century_rate',
            'weather_score', 'pitch_batting_score', 'innings_pressure', 'session_score'
        ]
        
        interaction_features = [
            'quality_vs_conditions', 'form_venue_interaction', 
            'experience_pressure', 'form_vs_opposition'
        ]
        
        # Combine all features
        selected_features = basic_features + form_features + context_features + interaction_features
        
        # Filter features that exist in the dataframe
        available_features = [f for f in selected_features if f in df.columns]
        
        return df[available_features + [target_col] if target_col in df.columns else available_features]
    
    def scale_features(self, df, feature_cols=None, fit=True):
        """Scale numerical features"""
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns
        
        scaled_df = df.copy()
        
        for col in feature_cols:
            if col in df.columns:
                if fit:
                    if col not in self.scalers:
                        self.scalers[col] = StandardScaler()
                    scaled_df[col] = self.scalers[col].fit_transform(df[[col]])
                else:
                    if col in self.scalers:
                        scaled_df[col] = self.scalers[col].transform(df[[col]])
        
        return scaled_df
    
    def get_feature_importance_names(self):
        """Get names of engineered features with descriptions"""
        return {
            'career_average': 'Career batting average',
            'career_centuries': 'Total career centuries',
            'form_score': 'Recent form score (0-1)',
            'consistency_score': 'Batting consistency score',
            'momentum_score': 'Current momentum score',
            'venue_avg_runs': 'Average runs at venue',
            'venue_century_rate': 'Century rate at venue',
            'opp_avg_runs': 'Average runs vs opposition',
            'opp_century_rate': 'Century rate vs opposition',
            'weather_score': 'Weather favorability score',
            'pitch_batting_score': 'Pitch batting difficulty score',
            'innings_pressure': 'Batting pressure score',
            'quality_vs_conditions': 'Player quality vs conditions',
            'form_venue_interaction': 'Form x venue familiarity',
            'experience_pressure': 'Experience vs pressure situation'
        }
