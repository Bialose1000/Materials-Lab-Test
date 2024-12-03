import pandas as pd
import numpy as np

# Load the data from the CSV file, skipping the first two rows
filename = "Modulus_Steel.csv" 
data = pd.read_csv(filename, skiprows=2)

# Ensure columns are named correctly
data.columns = ["time", "displacement_mm", "force_kN", "strain"]

# Convert displacement from mm to meters, and force from kN to N
data['displacement'] = data['displacement_mm'] / 1000  # mm to meters
data['force'] = data['force_kN'] * 1000  # kN to N

# Calculate stress (assuming cross-sectional area A is given)
cross_sectional_area = 19.625 * 1e-6
data['stress'] = data['force'] / cross_sectional_area

# Perform linear regression to find the slope (Young's modulus)
coefficients = np.polyfit(data['strain'], data['stress'], 1)
youngs_modulus = coefficients[0]

# Calculate standard error of the slope
y_fit = np.polyval(coefficients, data['strain'])  # Fitted stress values
residuals = data['stress'] - y_fit  # Residuals
n = len(data['strain'])  # Number of data points
p = 2  # Number of parameters in the model (slope and intercept)
rss = np.sum(residuals**2)  # Residual sum of squares
stderr_slope = np.sqrt(rss / (n - p)) / np.sqrt(np.sum((data['strain'] - np.mean(data['strain']))**2))

# Output the results
print(f"Young's modulus: {youngs_modulus:.2e} Pa")
print(f"Standard error of Young's modulus: {stderr_slope:.2e} Pa")

# Optionally save results to a new CSV file
# data.to_csv("elastic_region_data_with_modulus.csv", index=False)
