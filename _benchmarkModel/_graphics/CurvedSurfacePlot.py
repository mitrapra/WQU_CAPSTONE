import matplotlib.pyplot as plt  # Import matplotlib for plotting
import mpl_toolkits

from mpl_toolkits.mplot3d import Axes3D  # Import Axes3D for 3D plotting

import numpy as np  # Import numpy for numerical operations
import scipy

from scipy.interpolate import griddata  # Import griddata for interpolating scattered data

class CurvedSurfacePlot:
    def __init__(self, CAGR_values, ALPHA_values, STABILITY_values):
        """
        Initialize CurvedSurfacePlot class with lists of values for plotting.

        Args:
            CAGR_values (list):      List of Compound Annual Growth Rate (CAGR) values for the Z axis.
            ALPHA_values (list):     List of ALPHA values (a financial indicator) for the X axis.
            STABILITY_values (list): List of STABILITY values (a measure of volatility) for the Y axis.
        """
        self.CAGR_values      = CAGR_values  # Store CAGR values
        self.ALPHA_values     = ALPHA_values  # Store ALPHA values
        self.STABILITY_values = STABILITY_values  # Store STABILITY values

    def plot_surface(self, filename: bool=False):
        """
        Plot a 3D curved surface based on the initialized values.
        """
        # Create a meshgrid for the X and Y axes for a smooth surface plot
        ALPHA_values_grid, STABILITY_values_grid = np.meshgrid(
            np.linspace(min(self.ALPHA_values), max(self.ALPHA_values), 100),  # X axis grid from min to max ALPHA
            np.linspace(min(self.STABILITY_values), max(self.STABILITY_values), 100)  # Y axis grid from min to max STABILITY
        )

        # Interpolate the CAGR values to create a smooth surface over the grid
        # This uses cubic interpolation to estimate CAGR values across the grid
        CAGR_values_smooth_grid = griddata(
            (self.ALPHA_values, self.STABILITY_values),  # Original points
            self.CAGR_values,                            # Original CAGR values
            (ALPHA_values_grid, STABILITY_values_grid),  # Grid to interpolate on
            method='cubic'                               # Cubic interpolation method
        )

        # Create a figure for plotting
        fig = plt.figure(figsize=(16, 12))           # Set figure size
        ax  = fig.add_subplot(111, projection='3d')  # Add a 3D subplot
        ax.set_box_aspect(aspect=None, zoom=0.85)	 # Control the viewing Zoom

        # Plot the surface using the interpolated grid of CAGR values
        ax.plot_surface(
            ALPHA_values_grid, STABILITY_values_grid, CAGR_values_smooth_grid,
            cmap='viridis'  # Color map for the surface
        )

        # Set axis labels and plot title
        ax.set_xlabel('ALPHA')  # X axis label
        ax.set_ylabel('STABILITY')  # Y axis label
        ax.set_zlabel('CAGR', labelpad=15)  # Z axis label
        ax.set_title('3D Diagnostic Plot')  # Plot title
        
        # Save the Figure
        if filename:
            plt.savefig("Data/GeometricOptimalPortfolio.png")

        # Display the plot
        plt.show()