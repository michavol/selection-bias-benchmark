# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax
from scipy.stats import zscore

# %% [markdown]
# ### 1. Create Data

# %%
import numpy as np

## SETTINGS ##
# toy_example = "ex1"
propensity_scale = 100
num_points_grid = 10000
num_points = 200

toy_example = "ex2_nonlinear"
# ex1: high bias, high total bias, high outcome bias
# ex2: high bias, high total bias, low Y0, Y1 bias, high Y1-Y0 bias


## 1. PATIENT DATA ##

# Calculate the number of points along each aX_gridis
points_per_axis = int(np.sqrt(num_points_grid))

k = 10
logistic = lambda x: 1 / (1 + np.exp(-k * (x - 0.5))) #logistic
nonlinearity = logistic

# Generate linearly spaced points along each axis
x0_grid = np.linspace(0, 1, points_per_axis)
x1_grid = np.linspace(0, 1, points_per_axis)

# Sample numpy array from uniform distribution
x0 = np.random.uniform(0, 1, num_points)
x1 = np.random.uniform(0, 1, num_points)

# Create a meshgrid
xx0_grid, xx1_grid = np.meshgrid(x0_grid, x1_grid)

# Flatten the meshgrid to create a 1000x1 array
X_grid = np.vstack([xx0_grid.ravel(), xx1_grid.ravel()]).T
X = np.vstack([x0.ravel(), x1.ravel()]).T
X0_grid = X_grid[:, 0]
X1_grid = X_grid[:, 1]
X0 = X[:, 0]
X1 = X[:, 1]


## 2. OUTCOMES ##
# Create Y_grid
if toy_example.startswith("ex1"):
    Y_grid = np.array([X0_grid,1-X0_grid]).T
    Y = np.array([X0,1-X0]).T

elif toy_example.startswith("ex2"):
    Y_grid = np.array([X0_grid,1-X1_grid]).T
    Y = np.array([X0,1-X1]).T

if toy_example.endswith("nonlinear"):
    Y = nonlinearity(Y)
    Y_grid = nonlinearity(Y_grid)

Y0 = Y[:, 0]
Y1 = Y[:, 1]

## 3. TREATMENT ##
if toy_example.startswith("ex1"):
    scores_grid = np.array([X0_grid,1-X0_grid]).T
    scores = np.array([X0,1-X0]).T

    scores_grid = zscore(scores_grid, axis=0)
    scores = zscore(scores, axis=0)

elif toy_example.startswith("ex2"):
    scores_grid = np.array([X0_grid,1-X1_grid]).T
    scores = np.array([X0,1-X1]).T

    scores_grid = zscore(scores_grid, axis=0)
    scores = zscore(scores, axis=0)

# Apply the softmax function to each row to get probabilities
p_grid = softmax(propensity_scale*scores_grid, axis=1)
p = softmax(propensity_scale*scores, axis=1)

# Make sure rows add up to one again
row_sums_grid = p_grid.sum(axis=1, keepdims=True)
row_sums = p.sum(axis=1, keepdims=True)

p_grid = p_grid / row_sums_grid
p = p / row_sums

propensities_grid = p_grid
propensities = p

T = np.array([np.random.choice([tre for tre in range(propensities.shape[1])], p=row) for row in propensities])

# Create contourplot with X_grid and Y_grid
import matplotlib.pyplot as plt
import numpy as np

def plot_propensities(xx0_grid, xx1_grid, propensities_grid, points_per_axis):
    # Calculate the min and max values for the color scale
    vmin = min(propensities_grid[:, 0].min(), propensities_grid[:, 1].min())
    vmax = max(propensities_grid[:, 0].max(), propensities_grid[:, 1].max())

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)

    # Plot the first propensity
    contour0 = axs[0].contourf(xx0_grid, xx1_grid, propensities_grid[:, 0].reshape(points_per_axis, points_per_axis), alpha=0.4, levels=20, vmin=vmin, vmax=vmax, cmap='Spectral')
    axs[0].scatter(X[:,0], X[:,1], c=T, s=2, cmap='Spectral')
    axs[0].set_title('A0 Distribution')
    axs[0].set_aspect('equal')

    # Plot the second propensity
    contour1 = axs[1].contourf(xx0_grid, xx1_grid, propensities_grid[:, 1].reshape(points_per_axis, points_per_axis), alpha=0.4, levels=20, vmin=vmin, vmax=vmax, cmap='Spectral')
    # plt.scatter(X, c=T, alpha=0.5)
    axs[1].scatter(X[:,0], X[:,1], c=T, s=2, cmap='Spectral')

    axs[1].set_title('A1 Distribution')
    axs[1].set_aspect('equal')

    # Add a single shared colorbar
    cbar = fig.colorbar(contour1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)

    plt.show()
    
def plot_outcomes(xx0_grid, xx1_grid, X0, X1, Y_grid, T, points_per_axis):
    # Calculate the min and max values for the color scale
    vmin = min(Y_grid[:, 0].min(), Y_grid[:, 1].min(), (Y_grid[:, 1] - Y_grid[:, 0]).min())
    vmax = max(Y_grid[:, 0].max(), Y_grid[:, 1].max(), (Y_grid[:, 1] - Y_grid[:, 0]).max())

    # Plot the above three plots in a single figure next to each other, all with the same colorbar
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), sharex=True, sharey=True)
    contour0 = axs[0].contourf(xx0_grid, xx1_grid, Y_grid[:, 0].reshape(points_per_axis, points_per_axis), alpha=0.5, levels=20, vmin=vmin, vmax=vmax)
    axs[0].scatter(X0, X1, c=T, s=2, cmap='Spectral')
    axs[0].set_title('Y0')
    axs[0].set_aspect('equal')

    contour1 = axs[1].contourf(xx0_grid, xx1_grid, Y_grid[:, 1].reshape(points_per_axis, points_per_axis), alpha=0.5, levels=20, vmin=vmin, vmax=vmax)
    axs[1].scatter(X0, X1, c=T, s=2, cmap='Spectral')
    axs[1].set_title('Y1')
    axs[1].set_aspect('equal')

    contour_diff = axs[2].contourf(xx0_grid, xx1_grid, (Y_grid[:, 1] - Y_grid[:, 0]).reshape(points_per_axis, points_per_axis), alpha=0.5, levels=20, vmin=vmin, vmax=vmax)
    axs[2].scatter(X0, X1, c=T, s=2, cmap='Spectral')
    axs[2].set_title('Y1-Y0')
    axs[2].set_aspect('equal')

    # Add a single shared colorbar for all three plots
    cbar = fig.colorbar(contour_diff, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)

    plt.show()

def plot_outcome_dists(Y0, Y1, T):
    # Calculate Y1 - Y0
    Y_diff = Y1 - Y0

    # Create subplots
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # Define colors for different values of T
    # colors = ['blue', 'orange']
    # Get extreme colors of spectral colormap
    colors = plt.cm.Spectral(np.linspace(0, 1, len(np.unique(T))))

    # Plot the distribution of Y0 colored by T
    for t in np.unique(T):
        axs[0].hist(Y0[T == t], bins=30, color=colors[t], alpha=0.7, label=f'T={t}')
    axs[0].set_title('Distribution of Y0')
    axs[0].set_xlabel('Y0')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()

    # Plot the distribution of Y1 colored by T
    for t in np.unique(T):
        axs[1].hist(Y1[T == t], bins=30, color=colors[t], alpha=0.7, label=f'T={t}')
    axs[1].set_title('Distribution of Y1')
    axs[1].set_xlabel('Y1')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()

    # Plot the distribution of Y1 - Y0 colored by T
    for t in np.unique(T):
        axs[2].hist(Y_diff[T == t], bins=30, color=colors[t], alpha=0.7, label=f'T={t}')
    axs[2].set_title('Distribution of Y1 - Y0')
    axs[2].set_xlabel('Y1 - Y0')
    axs[2].set_ylabel('Frequency')
    axs[2].legend()

    # Plot the joint distribution of Y0 and Y1 colored by T
    for t in np.unique(T):
        axs[3].scatter(Y0[T == t], Y1[T == t], color=colors[t], s=8, label=f'T={t}', alpha=0.7)
    axs[3].set_title('Joint Distribution of Y0 and Y1')
    axs[3].set_xlabel('Y0')
    axs[3].set_ylabel('Y1')
    axs[3].legend()

    # plt.tight_layout()
    plt.show()

plot_propensities(xx0_grid, xx1_grid, propensities_grid, points_per_axis)
plot_outcomes(xx0_grid, xx1_grid, X0, X1, Y_grid, T, points_per_axis)
plot_outcome_dists(Y0, Y1, T)

# %% [markdown]
# ### 2. Plotting


