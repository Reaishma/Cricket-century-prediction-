import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Generate a larger dataset
np.random.seed(42)
data = {
    'Runs': np.random.randint(0, 200, 100),
    'Strike Rate': np.random.randint(50, 200, 100),
    'Balls Faced': np.random.randint(10, 200, 100),
    'Fours': np.random.randint(0, 20, 100),
    'Sixes': np.random.randint(0, 10, 100),
    'Century': np.where(np.random.randint(0, 200, 100) > 100, 1, 0)
}

df = pd.DataFrame(data)

# Create new features
df['Run Rate'] = df['Runs'] / df['Balls Faced']
df['Boundary Percentage'] = ((df['Fours'] * 4) + (df['Sixes'] * 6)) / df['Runs']
df['Dot Ball Percentage'] = (df['Balls Faced'] - (df['Fours'] + df['Sixes'])) / df['Balls Faced']
df['Run Rate vs Strike Rate Ratio'] = df['Run Rate'] / (df['Strike Rate'] / 100)
df['Boundary to Dot Ball Ratio'] = (df['Fours'] + df['Sixes']) / (df['Balls Faced'] - (df['Fours'] + df['Sixes']))
df['Runs per Ball Faced'] = df['Runs'] / df['Balls Faced']

# Define features and target
X = df[['Runs', 'Strike Rate', 'Run Rate', 'Boundary Percentage', 'Dot Ball Percentage', 'Run Rate vs Strike Rate Ratio', 'Boundary to Dot Ball Ratio', 'Runs per Ball Faced']]
y = df['Century']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))