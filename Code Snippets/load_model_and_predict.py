import joblib
import pandas as pd

# Load the trained model
model = joblib.load('accident_severity_model.pkl')

# Create a sample input for prediction
sample_data = pd.DataFrame({
    'Weather_Conditions': [2],       # 1=Clear, 2=Rainy, 3=Foggy
    'Road_Surface_Conditions': [1],  # 1=Dry, 2=Wet
    'Light_Conditions': [2],         # 1=Daylight, 2=Night
    'Vehicle_Speed': [85],
    'Driver_Age': [29],
    'Road_Type': [3],                # 3=Highway
    'Number_of_Vehicles': [2],
    'Time_of_Day': [21]              # 21=9PM
})

# Predict accident severity
prediction = model.predict(sample_data)
print("Predicted Accident Severity:", prediction[0])
