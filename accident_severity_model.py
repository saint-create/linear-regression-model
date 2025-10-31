# accident_severity_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset (replace with your actual file path)
data = pd.read_csv("road_accidents.csv")

# Select relevant columns
X = data[['Weather_Conditions', 'Road_Surface_Conditions', 'Light_Conditions',
          'Vehicle_Speed', 'Driver_Age', 'Road_Type', 'Number_of_Vehicles', 'Time_of_Day']]
y = data['Accident_Severity']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Save the model
joblib.dump(model, 'accident_severity_model.pkl')
print("Model saved successfully!")