# test_voronoi.py
# (Translation of test_voronoi.m)

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def run_test_voronoi():
    plt.close('all')
    print("--- Running test_voronoi.py ---")

    # Hardcoded centroids from the MATLAB file
    ctrs = np.array([
        [-1.0127,  1.6371],
        [ 1.6087,  0.0899],
        [ 1.3949, -1.4789],
        [-1.5454, -0.9398],
        [-0.3967, -0.3850],
        [-0.3551, -1.6383],
        [ 0.8018,  1.3046],
        [-0.1245,  0.6019],
        [-1.3861,  0.3074],
        [ 0.6694, -0.4262]
    ])

    # Create figure
    fig, ax = plt.subplots()

    # Calculate and plot the Voronoi diagram
    # scipy.spatial.Voronoi calculates the geometry
    vor = Voronoi(ctrs)
    
    # scipy.spatial.voronoi_plot_2d is the equivalent of MATLAB's plot
    # It draws both the points and the tessellation lines
    voronoi_plot_2d(vor, ax=ax)

    # Set axis to be equal, as in MATLAB's 'axis equal'
    ax.set_aspect('equal')
    
    # Save the figure
    output_filename = 'voronoi_py_test.png'
    plt.savefig(output_filename)
    print(f"Saved plot to '{output_filename}'")

    # Show the plot
    plt.title("Test Voronoi Diagram (Python)")
    plt.show()

if __name__ == '__main__':
    run_test_voronoi()