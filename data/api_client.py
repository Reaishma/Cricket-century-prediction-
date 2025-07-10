import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import numpy as np

class CricketAPIClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('CRICKET_API_KEY', 'demo_key')
        self.base_url = "https://api.cricapi.com/v1"
        self.headers = {
            'apikey': self.api_key,
            'Content-Type': 'application/json'
        }
        
    def get_current_matches(self):
        """Get current/upcoming matches"""
        try:
            url = f"{self.base_url}/currentMatches"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                # Return mock data if API fails
                return self._get_mock_current_matches()
                
        except Exception as e:
            print(f"Error fetching current matches: {e}")
            return self._get_mock_current_matches()
    
    def get_player_stats(self, player_id):
        """Get player statistics"""
        try:
            url = f"{self.base_url}/players/{player_id}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                # Return mock data if API fails
                return self._get_mock_player_stats(player_id)
                
        except Exception as e:
            print(f"Error fetching player stats: {e}")
            return self._get_mock_player_stats(player_id)
    
    def get_match_details(self, match_id):
        """Get detailed match information"""
        try:
            url = f"{self.base_url}/matches/{match_id}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                # Return mock data if API fails
                return self._get_mock_match_details(match_id)
                
        except Exception as e:
            print(f"Error fetching match details: {e}")
            return self._get_mock_match_details(match_id)
    
    def get_team_stats(self, team_name):
        """Get team statistics"""
        try:
            url = f"{self.base_url}/teams"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                teams = response.json()
                for team in teams.get('data', []):
                    if team.get('name', '').lower() == team_name.lower():
                        return team
                return None
            else:
                # Return mock data if API fails
                return self._get_mock_team_stats(team_name)
                
        except Exception as e:
            print(f"Error fetching team stats: {e}")
            return self._get_mock_team_stats(team_name)
    
    def get_live_score(self, match_id):
        """Get live score for a match"""
        try:
            url = f"{self.base_url}/matches/{match_id}/live"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                # Return mock data if API fails
                return self._get_mock_live_score(match_id)
                
        except Exception as e:
            print(f"Error fetching live score: {e}")
            return self._get_mock_live_score(match_id)
    
    def search_players(self, query):
        """Search for players"""
        try:
            url = f"{self.base_url}/players"
            params = {'search': query}
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                # Return mock data if API fails
                return self._get_mock_player_search(query)
                
        except Exception as e:
            print(f"Error searching players: {e}")
            return self._get_mock_player_search(query)
    
    def get_weather_data(self, venue):
        """Get weather data for a venue"""
        try:
            # This would typically use a weather API
            # For now, return mock weather data
            return self._get_mock_weather_data(venue)
            
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return self._get_mock_weather_data(venue)
    
    def get_pitch_report(self, venue):
        """Get pitch report for a venue"""
        try:
            # This would typically use a cricket data API
            # For now, return mock pitch data
            return self._get_mock_pitch_report(venue)
            
        except Exception as e:
            print(f"Error fetching pitch report: {e}")
            return self._get_mock_pitch_report(venue)
    
    # Mock data methods for fallback
    def _get_mock_current_matches(self):
        """Return mock current matches data"""
        return {
            "data": [
                {
                    "id": "match_001",
                    "name": "India vs Australia",
                    "matchType": "ODI",
                    "status": "Live",
                    "venue": "Melbourne Cricket Ground",
                    "date": datetime.now().isoformat(),
                    "teams": ["India", "Australia"],
                    "teamInfo": [
                        {"name": "India", "shortname": "IND"},
                        {"name": "Australia", "shortname": "AUS"}
                    ]
                },
                {
                    "id": "match_002",
                    "name": "England vs Pakistan",
                    "matchType": "Test",
                    "status": "Upcoming",
                    "venue": "Lords",
                    "date": (datetime.now() + timedelta(days=1)).isoformat(),
                    "teams": ["England", "Pakistan"],
                    "teamInfo": [
                        {"name": "England", "shortname": "ENG"},
                        {"name": "Pakistan", "shortname": "PAK"}
                    ]
                }
            ]
        }
    
    def _get_mock_player_stats(self, player_id):
        """Return mock player stats"""
        players = {
            "virat_kohli": {
                "name": "Virat Kohli",
                "country": "India",
                "role": "Batsman",
                "stats": {
                    "matches": 254,
                    "runs": 12169,
                    "average": 57.32,
                    "centuries": 43,
                    "strikeRate": 93.17
                }
            },
            "babar_azam": {
                "name": "Babar Azam",
                "country": "Pakistan",
                "role": "Batsman",
                "stats": {
                    "matches": 83,
                    "runs": 4442,
                    "average": 56.83,
                    "centuries": 17,
                    "strikeRate": 88.57
                }
            }
        }
        return players.get(player_id, players["virat_kohli"])
    
    def _get_mock_match_details(self, match_id):
        """Return mock match details"""
        return {
            "id": match_id,
            "name": "India vs Australia",
            "matchType": "ODI",
            "status": "Live",
            "venue": "Melbourne Cricket Ground",
            "date": datetime.now().isoformat(),
            "teams": ["India", "Australia"],
            "score": [
                {"team": "India", "runs": 287, "wickets": 6, "overs": 50},
                {"team": "Australia", "runs": 156, "wickets": 4, "overs": 32.2}
            ],
            "players": [
                {"name": "Virat Kohli", "team": "India", "runs": 89, "balls": 94},
                {"name": "Steve Smith", "team": "Australia", "runs": 45, "balls": 67}
            ]
        }
    
    def _get_mock_team_stats(self, team_name):
        """Return mock team stats"""
        return {
            "name": team_name,
            "ranking": {
                "Test": np.random.randint(1, 10),
                "ODI": np.random.randint(1, 10),
                "T20": np.random.randint(1, 10)
            },
            "recentForm": ["W", "L", "W", "W", "L"],
            "stats": {
                "matches": np.random.randint(100, 500),
                "wins": np.random.randint(50, 300),
                "losses": np.random.randint(50, 200)
            }
        }
    
    def _get_mock_live_score(self, match_id):
        """Return mock live score"""
        return {
            "id": match_id,
            "status": "Live",
            "currentInnings": 2,
            "score": {
                "team": "Australia",
                "runs": 156,
                "wickets": 4,
                "overs": 32.2
            },
            "batsmen": [
                {"name": "Steve Smith", "runs": 45, "balls": 67, "fours": 4, "sixes": 1},
                {"name": "Glenn Maxwell", "runs": 23, "balls": 18, "fours": 2, "sixes": 1}
            ],
            "bowler": {
                "name": "Jasprit Bumrah",
                "overs": 6.2,
                "runs": 28,
                "wickets": 2
            }
        }
    
    def _get_mock_player_search(self, query):
        """Return mock player search results"""
        all_players = [
            {"id": "virat_kohli", "name": "Virat Kohli", "country": "India"},
            {"id": "babar_azam", "name": "Babar Azam", "country": "Pakistan"},
            {"id": "steve_smith", "name": "Steve Smith", "country": "Australia"},
            {"id": "kane_williamson", "name": "Kane Williamson", "country": "New Zealand"},
            {"id": "joe_root", "name": "Joe Root", "country": "England"}
        ]
        
        # Filter players based on query
        filtered_players = [
            player for player in all_players 
            if query.lower() in player["name"].lower()
        ]
        
        return {"data": filtered_players}
    
    def _get_mock_weather_data(self, venue):
        """Return mock weather data"""
        return {
            "venue": venue,
            "temperature": np.random.randint(15, 35),
            "humidity": np.random.randint(40, 80),
            "windSpeed": np.random.randint(5, 25),
            "conditions": np.random.choice(["Clear", "Overcast", "Light Rain", "Heavy Rain"]),
            "forecast": [
                {
                    "time": "Morning",
                    "temperature": np.random.randint(20, 30),
                    "conditions": "Clear"
                },
                {
                    "time": "Afternoon", 
                    "temperature": np.random.randint(25, 35),
                    "conditions": "Overcast"
                },
                {
                    "time": "Evening",
                    "temperature": np.random.randint(18, 28),
                    "conditions": "Clear"
                }
            ]
        }
    
    def _get_mock_pitch_report(self, venue):
        """Return mock pitch report"""
        return {
            "venue": venue,
            "pitchType": np.random.choice(["Flat", "Green", "Dusty", "Cracked"]),
            "averageScore": np.random.randint(180, 320),
            "paceFriendly": np.random.choice([True, False]),
            "spinFriendly": np.random.choice([True, False]),
            "battingFirst": {
                "advantage": np.random.choice(["High", "Medium", "Low"]),
                "averageScore": np.random.randint(160, 300)
            },
            "recentMatches": [
                {"team1": "India", "team2": "Australia", "score1": 287, "score2": 245},
                {"team1": "England", "team2": "Pakistan", "score1": 312, "score2": 289}
            ]
        }
    
    def get_player_form_data(self, player_name, last_n_matches=10):
        """Get player's recent form data"""
        try:
            # This would typically query the API for recent match data
            # For now, return mock form data
            return self._get_mock_player_form(player_name, last_n_matches)
            
        except Exception as e:
            print(f"Error fetching player form: {e}")
            return self._get_mock_player_form(player_name, last_n_matches)
    
    def _get_mock_player_form(self, player_name, last_n_matches):
        """Return mock player form data"""
        np.random.seed(hash(player_name) % 1000)
        
        matches = []
        for i in range(last_n_matches):
            match_date = datetime.now() - timedelta(days=i*7)
            runs = max(0, int(np.random.gamma(2, 15)))
            
            matches.append({
                "date": match_date.isoformat(),
                "opponent": np.random.choice(["Australia", "England", "Pakistan", "South Africa"]),
                "format": np.random.choice(["Test", "ODI", "T20"]),
                "runs": runs,
                "balls": runs + np.random.randint(10, 50),
                "notOut": np.random.choice([True, False], p=[0.2, 0.8]),
                "venue": np.random.choice(["MCG", "Lords", "Eden Gardens", "Wankhede"])
            })
        
        return {
            "player": player_name,
            "matches": matches,
            "summary": {
                "totalRuns": sum(m["runs"] for m in matches),
                "average": sum(m["runs"] for m in matches) / len(matches),
                "centuries": sum(1 for m in matches if m["runs"] >= 100),
                "fifties": sum(1 for m in matches if 50 <= m["runs"] < 100)
            }
        }
