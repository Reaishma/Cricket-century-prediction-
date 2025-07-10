import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
    def load_sample_data(self, n_samples=5000):
        """Generate sample cricket data for demonstration"""
        np.random.seed(42)
        
        # Player information
        players = [
            'Virat Kohli', 'Babar Azam', 'Steve Smith', 'Kane Williamson', 'Joe Root',
            'Rohit Sharma', 'David Warner', 'Quinton de Kock', 'Jos Buttler', 'AB de Villiers',
            'Chris Gayle', 'MS Dhoni', 'Eoin Morgan', 'Aaron Finch', 'Jonny Bairstow',
            'KL Rahul', 'Shikhar Dhawan', 'Jason Roy', 'Martin Guptill', 'Faf du Plessis'
        ]
        
        venues = [
            'MCG', 'Lords', 'Eden Gardens', 'Wankhede', 'SCG', 'The Oval', 'Old Trafford',
            'Headingley', 'Basin Reserve', 'Wanderers', 'Newlands', 'Galle', 'Pallekele',
            'Dubai', 'Sharjah', 'Abu Dhabi', 'Trent Bridge', 'Edgbaston', 'Cardiff', 'Southampton'
        ]
        
        teams = [
            'India', 'Pakistan', 'Australia', 'England', 'New Zealand', 'South Africa',
            'West Indies', 'Sri Lanka', 'Bangladesh', 'Afghanistan', 'Zimbabwe', 'Ireland'
        ]
        
        # Generate sample data
        data = []
        for i in range(n_samples):
            # Basic match info
            player = np.random.choice(players)
            venue = np.random.choice(venues)
            team = np.random.choice(teams)
            opposition = np.random.choice([t for t in teams if t != team])
            match_format = np.random.choice(['Test', 'ODI', 'T20'], p=[0.3, 0.5, 0.2])
            
            # Date (last 5 years)
            start_date = datetime.now() - timedelta(days=365*5)
            random_days = np.random.randint(0, 365*5)
            match_date = start_date + timedelta(days=random_days)
            
            # Player statistics
            career_matches = np.random.randint(50, 300)
            career_runs = np.random.randint(2000, 15000)
            career_average = career_runs / career_matches + np.random.normal(0, 5)
            career_centuries = np.random.randint(10, 50)
            
            # Recent form (last 10 matches)
            recent_runs = np.random.randint(200, 800)
            recent_average = recent_runs / 10 + np.random.normal(0, 3)
            recent_centuries = np.random.randint(0, 4)
            
            # Venue specific stats
            venue_matches = np.random.randint(5, 30)
            venue_runs = np.random.randint(200, 2000)
            venue_average = venue_runs / venue_matches + np.random.normal(0, 4)
            venue_centuries = np.random.randint(0, 8)
            
            # Opposition record
            vs_opp_matches = np.random.randint(10, 50)
            vs_opp_runs = np.random.randint(400, 3000)
            vs_opp_average = vs_opp_runs / vs_opp_matches + np.random.normal(0, 4)
            vs_opp_centuries = np.random.randint(0, 12)
            
            # Weather and pitch conditions
            weather = np.random.choice(['Clear', 'Overcast', 'Light Rain', 'Heavy Rain'], p=[0.5, 0.3, 0.15, 0.05])
            temperature = np.random.normal(25, 8)
            humidity = np.random.normal(60, 20)
            wind_speed = np.random.gamma(2, 5)
            
            pitch_type = np.random.choice(['Flat', 'Green', 'Dusty', 'Cracked'], p=[0.4, 0.25, 0.25, 0.1])
            
            # Match context
            innings = np.random.choice([1, 2])
            batting_position = np.random.randint(1, 8)
            target_score = np.random.randint(150, 400) if innings == 2 else None
            
            # Time factors
            is_day_night = np.random.choice([0, 1], p=[0.7, 0.3])
            session = np.random.choice(['Morning', 'Afternoon', 'Evening'])
            
            # Calculate century probability based on features
            base_prob = 0.1  # Base probability of scoring a century
            
            # Adjust based on player quality (using career average as proxy)
            if career_average > 45:
                base_prob *= 1.5
            elif career_average > 35:
                base_prob *= 1.2
            
            # Adjust based on recent form
            if recent_average > 40:
                base_prob *= 1.3
            elif recent_average < 20:
                base_prob *= 0.7
            
            # Adjust based on venue familiarity
            if venue_average > career_average + 5:
                base_prob *= 1.2
            elif venue_average < career_average - 5:
                base_prob *= 0.8
            
            # Adjust based on opposition
            if vs_opp_average > career_average + 3:
                base_prob *= 1.1
            elif vs_opp_average < career_average - 3:
                base_prob *= 0.9
            
            # Adjust based on conditions
            if weather == 'Clear' and pitch_type == 'Flat':
                base_prob *= 1.2
            elif weather == 'Heavy Rain' or pitch_type == 'Green':
                base_prob *= 0.8
            
            # Adjust based on batting position
            if batting_position <= 3:
                base_prob *= 1.1
            elif batting_position >= 6:
                base_prob *= 0.7
            
            # Generate target variable
            century = np.random.binomial(1, min(base_prob, 0.4))
            
            # Generate actual score
            if century:
                score = np.random.randint(100, 200)
            else:
                score = max(0, int(np.random.gamma(2, 15)))
            
            # Append to data
            data.append({
                'player': player,
                'venue': venue,
                'team': team,
                'opposition': opposition,
                'format': match_format,
                'date': match_date,
                'career_matches': career_matches,
                'career_runs': career_runs,
                'career_average': career_average,
                'career_centuries': career_centuries,
                'recent_runs': recent_runs,
                'recent_average': recent_average,
                'recent_centuries': recent_centuries,
                'venue_matches': venue_matches,
                'venue_runs': venue_runs,
                'venue_average': venue_average,
                'venue_centuries': venue_centuries,
                'vs_opp_matches': vs_opp_matches,
                'vs_opp_runs': vs_opp_runs,
                'vs_opp_average': vs_opp_average,
                'vs_opp_centuries': vs_opp_centuries,
                'weather': weather,
                'temperature': temperature,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'pitch_type': pitch_type,
                'innings': innings,
                'batting_position': batting_position,
                'target_score': target_score if target_score else 0,
                'is_day_night': is_day_night,
                'session': session,
                'score': score,
                'century': century
            })
        
        return pd.DataFrame(data)
    
    def clean_data(self, df):
        """Clean and preprocess the data"""
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = self.imputer.fit_transform(df[numeric_columns])
        
        # Handle categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna('Unknown')
        
        # Remove outliers using IQR method
        for col in numeric_columns:
            if col not in ['century', 'score']:  # Don't remove outliers from target variables
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features"""
        categorical_columns = ['player', 'venue', 'team', 'opposition', 'format', 
                              'weather', 'pitch_type', 'session']
        
        encoded_df = df.copy()
        
        for col in categorical_columns:
            if col in df.columns:
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    encoded_df[col] = self.label_encoders[col].fit_transform(df[col])
                else:
                    if col in self.label_encoders:
                        # Handle unknown categories
                        unique_values = set(df[col].unique())
                        known_values = set(self.label_encoders[col].classes_)
                        unknown_values = unique_values - known_values
                        
                        if unknown_values:
                            # Add unknown values to the encoder
                            all_values = list(known_values) + list(unknown_values)
                            self.label_encoders[col].classes_ = np.array(all_values)
                        
                        encoded_df[col] = self.label_encoders[col].transform(df[col])
        
        return encoded_df
    
    def create_time_features(self, df):
        """Create time-based features"""
        if 'date' in df.columns:
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_year'] = df['date'].dt.dayofyear
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features"""
        # Player-venue interaction
        if 'venue_average' in df.columns and 'career_average' in df.columns:
            df['venue_vs_career_avg'] = df['venue_average'] - df['career_average']
        
        # Recent form vs career form
        if 'recent_average' in df.columns and 'career_average' in df.columns:
            df['recent_vs_career_avg'] = df['recent_average'] - df['career_average']
        
        # Opposition specific performance
        if 'vs_opp_average' in df.columns and 'career_average' in df.columns:
            df['vs_opp_vs_career_avg'] = df['vs_opp_average'] - df['career_average']
        
        # Weather-pitch interaction
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['heat_index'] = df['temperature'] + (df['humidity'] / 100) * 10
        
        return df
    
    def prepare_features(self, df, target_column='century'):
        """Prepare features for model training"""
        # Make a copy to avoid modifying original
        processed_df = df.copy()
        
        # Clean data
        processed_df = self.clean_data(processed_df)
        
        # Create time features
        processed_df = self.create_time_features(processed_df)
        
        # Create interaction features
        processed_df = self.create_interaction_features(processed_df)
        
        # Encode categorical features
        processed_df = self.encode_categorical_features(processed_df, fit=True)
        
        # Separate features and target
        if target_column in processed_df.columns:
            X = processed_df.drop([target_column, 'score', 'date'], axis=1, errors='ignore')
            y = processed_df[target_column]
        else:
            X = processed_df.drop(['date'], axis=1, errors='ignore')
            y = None
        
        return X, y
    
    def get_feature_names(self):
        """Get the names of all features"""
        sample_df = self.load_sample_data(n_samples=100)
        X, _ = self.prepare_features(sample_df)
        return X.columns.tolist()
    
    def save_preprocessors(self, filepath):
        """Save preprocessing objects"""
        import joblib
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'imputer': self.imputer
        }, f"{filepath}_preprocessors.pkl")
    
    def load_preprocessors(self, filepath):
        """Load preprocessing objects"""
        import joblib
        preprocessors = joblib.load(f"{filepath}_preprocessors.pkl")
        self.label_encoders = preprocessors['label_encoders']
        self.scaler = preprocessors['scaler']
        self.imputer = preprocessors['imputer']
