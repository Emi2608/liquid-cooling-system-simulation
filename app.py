
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

# Set page title and description
st.title("🌡️ AI-Powered Liquid Cooling System Simulation")
st.write("""
This simulation models a liquid cooling system for a data center server, predicting the required cooling power based on temperature readings from an ESP32 sensor (in °F). 
The system circulates coolant through server racks, absorbing heat and transferring it to a heat exchanger, where it’s dissipated. 
Adjust the temperature slider to see the predicted cooling power and its behavior across a range of temperatures.
""")

# Load the trained model and scalers
model = load_model('cooling_model.h5')
input_scaler = joblib.load('input_scaler.pkl')
output_scaler = joblib.load('output_scaler.pkl')

# Temperature input slider
temp_input = st.slider(
    "Set Temperature (°F)",
    min_value=50,
    max_value=120,
    value=75,
    help="Simulate the temperature reading from an ESP32 sensor."
)

# Scale the input temperature using the training scaler
temp_array = input_scaler.transform(np.array([[temp_input]]))

# Predict cooling power
predicted_cooling_scaled = model.predict(temp_array, verbose=0)[0][0]
predicted_cooling = output_scaler.inverse_transform([[predicted_cooling_scaled]])[0][0]

# Display the prediction
st.metric(label="Predicted Cooling Power", value=f"{predicted_cooling:.2f} %")

# Generate data for visualization
temp_range = np.linspace(50, 120, 100).reshape(-1, 1)
temp_range_scaled = input_scaler.transform(temp_range)
predicted_cooling_scaled_range = model.predict(temp_range_scaled, verbose=0)
predicted_cooling_range = output_scaler.inverse_transform(predicted_cooling_scaled_range)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(temp_range, predicted_cooling_range, label='Predicted Cooling Power', color='blue')
plt.scatter([temp_input], [predicted_cooling], color='red', label='Current Input', zorder=5)
plt.xlabel('Temperature (°F)')
plt.ylabel('Cooling Power (%)')
plt.title('Cooling Power vs Temperature')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
st.pyplot(plt)

# Add a data center server image (using a placeholder URL)
try:
    st.image(
        'data_center.jpg',
        caption='Data Center Server',
        use_container_width=True
    )
except Exception as e:
    st.error(f"Failed to load image 'data_center.jpg': {str(e)}")
    st.write("Please ensure the image file is uploaded correctly.")
