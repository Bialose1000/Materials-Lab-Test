import pandas as pd
import numpy as np
from scipy.stats import linregress

initial_length = 0.025  # Initial length in meters
cross_sectional_area = 19.625 * 1e-6  # Cross-sectional area in square meters

# Load data from the CSV file
file_path = 'PMMA.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path, skiprows=9)

# Ensure the data has the expected columns
time = data.iloc[:, 0]
displacement_mm = data.iloc[:, 1]
displacement = displacement_mm / 1000  # Convert displacement to meters
force = data.iloc[:, 2]

# Calculate strain and stress
strain = displacement / initial_length
stress = force / cross_sectional_area

# Automatically detect the elastic region
# Use a sliding window to find the region with the best linear fit
window_size = 100
best_r2 = 0
best_slope = 0
best_intercept = 0
best_std_err = 0
elastic_start = 0
elastic_end = 0

for i in range(len(strain) - window_size):
    strain_window = strain[i:i + window_size]
    stress_window = stress[i:i + window_size]
    
    slope, intercept, r_value, p_value, std_err = linregress(strain_window, stress_window)
    if r_value**2 > best_r2:  # Look for the highest R^2 value
        best_r2 = r_value**2
        best_slope = slope
        best_intercept = intercept
        best_std_err = std_err
        elastic_start = i
        elastic_end = i + window_size

# Use the detected elastic region to calculate Young's modulus
strain_elastic = strain[elastic_start:elastic_end]
stress_elastic = stress[elastic_start:elastic_end]
youngs_modulus = best_slope

# Output the results
print(f"Young's Modulus: {youngs_modulus:.2e} Pa")
print(f"Standard Error: {best_std_err:.2e} Pa")
print(f"Elastic region detected between indices {elastic_start} and {elastic_end}")
