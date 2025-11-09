# help_sample.py
# (Translation of help_sample.m)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d

def run_help_sample():
    plt.close('all')
    print("--- Running help_sample.py ---")

    # Generate the 2D Gaussian data
    X1 = np.random.randn(5000, 2)
    X2 = 0.5 * np.random.randn(5000, 2)
    X3 = 0.2 * np.random.randn(5000, 2)
    X4 = 0.1 * np.random.randn(5000, 2)
    X = np.concatenate((X1, X2, X3, X4), axis=0)
    
    n_clusters = 30
    print(f"Running k-means with {n_clusters} clusters...")
    
    # Run k-means
    # n_init=1 (or 'auto') mimics MATLAB's default of a single run
    # random_state ensures the result is reproducible
    kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=42)
    kmeans.fit(X)
    ctrs = kmeans.cluster_centers_

    # Create figure
    fig, ax = plt.subplots()

    # Calculate the Voronoi diagram from the centroids
    vor = Voronoi(ctrs)

    # Plot the Voronoi diagram
    # We turn off 'show_points' so we can draw them ourselves
    # with the 'ko' (black circle) style from the MATLAB script.
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False)
    
    # Plot the centroids as 'ko' (black circles)
    # 'markerfacecolor="none"' makes the 'o' hollow
    ax.plot(ctrs[:, 0], ctrs[:, 1], 'ko', 
            markersize=12, linewidth=2, markerfacecolor='none')

    # Set axis limits and aspect ratio
    ax.axis([-2.2, 2.2, -2.2, 2.2])
    ax.set_aspect('equal')

    # Save the figure
    output_filename = 'voronoi_py_help.png'
    plt.savefig(output_filename)
    print(f"Saved plot to '{output_filename}'")
    
    # Show the plot
    plt.title("k-means Centroids and Voronoi Diagram (Python)")
    plt.show()

if __name__ == '__main__':
    run_help_sample()