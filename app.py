from flask import Flask, render_template
import numpy as np
import folium

app = Flask(__name__)

@app.route('/')
def index():
    # Example anchor positions and TDOA measurements
    anchor_positions = np.array([[0, 0], [1, 0], [0, 1]])
    tdoa_measurements = np.array([0.001, 0.002])  # Example TDOA measurements in seconds
    speed_of_sound = 343  # Speed of sound in air in meters per second

    # Perform TDOA multilateration
    estimated_position = tdoa_multilateration(anchor_positions, tdoa_measurements, speed_of_sound)

    # Plot on map
    map = folium.Map(location=[estimated_position[0], estimated_position[1]], zoom_start=15)
    folium.Marker(location=[estimated_position[0], estimated_position[1]], popup='Estimated Position').add_to(map)

    return render_template('index.html', map=map._repr_html_())

def tdoa_multilateration(anchor_positions, tdoa_measurements, speed_of_sound):
    """
    Perform TDOA multilateration for LoRa gateways.
    
    Args:
    - anchor_positions: Array of shape (n, 2) representing x, y coordinates of anchor points (gateways)
    - tdoa_measurements: Array of shape (n-1,) representing time difference of arrival measurements
    - speed_of_sound: Speed of sound in the medium
    
    Returns:
    - estimated_position: Estimated x, y coordinates of the target
    """
    n = anchor_positions.shape[0]
    
    # Construct linear system of equations
    A = np.zeros((n - 1, 2))
    b = np.zeros((n - 1,))
    
    for i in range(n - 1):
        A[i, 0] = anchor_positions[i + 1, 0] - anchor_positions[0, 0]
        A[i, 1] = anchor_positions[i + 1, 1] - anchor_positions[0, 1]
        b[i] = (tdoa_measurements[i] * speed_of_sound)
    
    # Solve linear system
    solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # Estimated position
    estimated_position = anchor_positions[0] + solution
    
    return estimated_position

if __name__ == '__main__':
    app.run(debug=True)
