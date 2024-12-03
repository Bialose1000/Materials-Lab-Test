import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Constants
original_length = 30 *1e-3
cross_sectional_area = 100 * 1e-6  # mm^2 to m^2

# Load and clean the dataset
file_path = 'PMMA.csv'
data = pd.read_csv(file_path, skiprows=21, names=["Time (s)", "Displacement (mm)", "Force (N)"])

# Convert columns to numeric
data["Displacement (mm)"] = pd.to_numeric(data["Displacement (mm)"], errors='coerce')

# Calculate Stress (Force / Area) and Strain (Displacement / Original Length)
data["Stress (Pa)"] = data["Force (N)"] / cross_sectional_area
data["Strain"] = (data["Displacement (mm)"] * 1e-3) / original_length  # Convert mm to m

# Determine Ultimate Stress (maximum stress)
ultimate_stress = data["Stress (Pa)"].max()
ultimate_strain = data.loc[data["Stress (Pa)"].idxmax(), "Strain"]

# Percent Elongation at Failure
percent_elongation = (data["Displacement (mm)"].iloc[-1] * 1e-3 / original_length) * 100

# Calculate Yield Stress using the 0.2% Offset Method
# Extract the elastic region (first linear portion) for slope calculation
elastic_region = data[data["Strain"] < 0.005]  # Assume elasticity within small strain range
x_elastic = elastic_region["Strain"]
y_elastic = elastic_region["Stress (Pa)"]

# Fit a linear line to the elastic region
def linear_fit(x, m, c):
    return m * x + c

params, _ = curve_fit(linear_fit, x_elastic, y_elastic)
slope = params[0]  # Slope of the elastic region (Young's Modulus)

# Generate the 0.2% offset line
offset_strain = data["Strain"] + 0.002  # 0.2% offset strain
offset_line = slope * offset_strain

# Find the intersection point (closest point where Stress-Offset Stress is minimum)
intersection_idx = np.argmin(np.abs(data["Stress (Pa)"] - offset_line))
yield_stress = data["Stress (Pa)"].iloc[intersection_idx]
yield_strain = data["Strain"].iloc[intersection_idx]

# Plot the Stress-Strain Curve and 0.2% Offset Line
plt.figure(figsize=(12, 8))

# Stress-Strain Curve
plt.plot(data["Strain"], data["Stress (Pa)"] / 1e6, label="Stress-Strain Curve", color="b", linewidth=2)

# Offset Line
plt.plot(offset_strain, offset_line / 1e6, '--', label="0.2% Offset Line", color="r", linewidth=1.5)

# Highlight Yield Stress
plt.scatter(yield_strain, yield_stress / 1e6, color='g', label=f"Yield Stress: {yield_stress/1e6:.2f} MPa")
plt.text(yield_strain + 0.002, yield_stress / 1e6, f"{yield_stress/1e6:.2f} MPa", color='g')

# Highlight Ultimate Stress
plt.scatter(ultimate_strain, ultimate_stress / 1e6, color='orange', label=f"Ultimate Stress: {ultimate_stress/1e6:.2f} MPa")
plt.text(ultimate_strain, ultimate_stress / 1e6 + 10, f"{ultimate_stress/1e6:.2f} MPa", color='orange')

final_strain = data["Strain"].iloc[-1]
plt.scatter(final_strain, data["Stress (Pa)"].iloc[-1] / 1e6, color='purple', label=f"Percent Elongation: {percent_elongation:.2f}%")
plt.text(final_strain, data["Stress (Pa)"].iloc[-1] / 1e6 - 10, f"{percent_elongation:.2f}%", color='purple')

# Scientific Layout
plt.title("Stress-Strain Graph for PMMA Sample", fontsize=16, fontweight='bold')
plt.xlabel("Strain (mm/mm)", fontsize=14)
plt.ylabel("Stress (MPa)", fontsize=14)
plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=12)

plt.xlim(0, data["Strain"].max() * 1.1)
plt.ylim(0, ultimate_stress / 1e6 * 1.2)  # Set reasonable limits to avoid compression

# Ensure better aspect ratio
plt.gca().set_aspect('auto', adjustable='box')

# Display the plot
plt.show()

# Results Summary
print(f"Ultimate Stress: {ultimate_stress:.2f} Pa")
print(f"Percent Elongation at Failure: {percent_elongation:.2f}%")
print(f"Yield Stress: {yield_stress:.2f} Pa")
