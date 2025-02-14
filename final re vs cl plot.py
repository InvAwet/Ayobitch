import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Given data
w_values = np.array([50, 100, 125, 150, 200, 100, 100])  # Width values
H_values = np.array([27, 27, 27, 27, 27, 50, 27])        # Height values
d_values = np.array([20, 20, 20, 20, 20, 20, 20])        # Depth values
cl_values = np.array([
    0.05146 * (np.exp(-0.5) / (H_values[0]/w_values[0])) - 0.0096,
    0.0374 * (np.exp(-0.5) / (H_values[1]/w_values[1])) - 0.0104,
    0.0323 * (np.exp(-0.5) / (H_values[2]/w_values[2])) - 0.01167,
    0.02158 * (np.exp(-0.5) / (H_values[3]/w_values[3])) - 0.00916,
    0.01637 * (np.exp(-0.5) / (H_values[4]/w_values[4])) - 0.011,
    0.3211 * (np.exp(-0.5)) - 0.0221,
    0.1366 * (np.exp(-0.5)) - 0.0099
])

# Combine input features (w, H, d) into a single array
X = np.column_stack((w_values, H_values, d_values))
y = cl_values

# Define the Gaussian Process Regressor with an RBF kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0, 1.0, 1.0], (1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Fit the model to the data
gpr.fit(X, y)

def calculate_cl(w_value, H_value, d_value):
    """
    Calculate the interpolated cl value for given w, H, and d using GPR.
    """
    # Ensure the input values are within the interpolation range
    if not (min(w_values) <= w_value <= max(w_values)) or \
       not (min(H_values) <= H_value <= max(H_values)) or \
       not (min(d_values) <= d_value <= max(d_values)):
        raise ValueError("Input values are outside the interpolation range.")
    
    # Predict the cl value using the trained GPR model
    cl_predicted = gpr.predict(np.array([[w_value, H_value, d_value]]))
    return cl_predicted[0]

# Example usage with predefined values
w_input = 100  # Example width value
H_input = 30   # Example height value
d_input = 20   # Example depth value

try:
    cl_output = calculate_cl(w_input, H_input, d_input)
    print(f"For w = {w_input}, H = {H_input}, d = {d_input}, the interpolated cl value is: {cl_output:.3f}")
except ValueError as e:
    print(e)

# Create a grid for w, H, and d for visualization
w_grid, H_grid = np.meshgrid(np.linspace(50, 200, 100), np.linspace(27, 50, 100))
d_grid = 20  # Constant depth

# Flatten the grid for prediction
X_grid = np.column_stack((w_grid.ravel(), H_grid.ravel(), np.full_like(w_grid.ravel(), d_grid)))

# Predict cl values for the grid
cl_grid = gpr.predict(X_grid).reshape(w_grid.shape)

# Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(w_grid, H_grid, cl_grid, cmap='viridis', edgecolor='none')

# Add labels and color bar
ax.set_xlabel('Width (w)')
ax.set_ylabel('Height (H)')
ax.set_zlabel('Coefficient of Lift (cl)')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='cl')

# Show the plot
plt.show()