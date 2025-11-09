# solution_ex2.py
#
# This script solves Exercise 2 by:
#  (1) Designing a 2D Vector Quantizer using k-means
#      (also known as the Generalized Lloyd Algorithm, or LBG).
#  (2) Drawing the Voronoi diagram for the resulting centroids
#      and explaining the result.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d

def run_exercise_2():
    plt.close('all')
    print("--- Running Exercise 2: 2D Vector Quantizer (k-means) & Voronoi ---")

    # --- (1) Design a 2D Vector Quantizer ---
    
    # --- Step 1: Generate 2D Data ---
    # We first need a 2D signal to quantize. This could represent
    # many things, e.g., (x, y) coordinates, (brightness, color)
    # of pixels, or two features from a speech signal.
    #
    # We create a "training set" of 12,000 vectors (points)
    # grouped into 4 distinct "clouds" or clusters.
    print("Generating 2D test data (12,000 vectors in 4 groups)...")
    X1 = np.random.randn(3000, 2) + 5 * np.ones((3000, 2))
    X2 = 2 * np.random.randn(3000, 2) + 0 * np.ones((3000, 2))
    X3 = 2 * np.random.randn(3000, 2) - 2 * np.ones((3000, 2))
    X4 = 3 * np.random.randn(3000, 2) - 5 * np.ones((3000, 2))
    X = np.concatenate((X1, X2, X3, X4), axis=0)

    # --- Step 2: Run k-means (Generalized Lloyd Algorithm) ---
    # We will design a quantizer with 5 levels (n_clusters = 5).
    # This is our "codebook" size. Note that we designed the data
    # with 4 groups, but we are *forcing* the quantizer to find 5.
    # This is a common test.
    
    n_clusters = 5
    print(f"Running k-means (Generalized Lloyd) to find {n_clusters} centroids...")
    
    # n_init=1 mimics MATLAB's 'Replicates', 1
    # random_state=42 makes the result reproducible
    kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=42)
    
    # 'fit_predict' runs the algorithm:
    # 1. 'fit': It finds the optimal 'centroid' locations.
    # 2. 'predict': It assigns each point in X to the nearest centroid.
    # 'idx' is the "quantized" index (0-4) for each vector in X.
    # 'centroid' is the "codebook" of our vector quantizer.
    
    idx = kmeans.fit_predict(X) 
    centroid = kmeans.cluster_centers_

    print("k-means complete. Centroids (Codebook) found.")

    # --- Step 3: Plot the k-means results ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each data point, colored by its assigned cluster
    colors = ['r', 'b', 'g', 'y', 'c']
    for i in range(n_clusters):
        cluster_points = X[idx == i]
        ax.plot(cluster_points[:, 0], cluster_points[:, 1], '.', 
                color=colors[i], markersize=8, label=f'Cluster {i+1}')

    # Plot the final centroids
    # These 5 points ARE the vector quantizer
    ax.plot(centroid[:, 0], centroid[:, 1], 'kx', 
            markersize=10, markeredgewidth=2, label='Centroids (Codebook)')
    ax.plot(centroid[:, 0], centroid[:, 1], 'ko', 
            markersize=10, markeredgewidth=2, markerfacecolor='none')

    # --- (2) Draw the Voronoi Diagram & Explain ---
    
    print("Calculating and drawing Voronoi diagram...")
    
    # A Voronoi diagram divides a plane into regions (cells)
    # based on a set of points (our centroids).
    #
    # **EXPLANATION:**
    # Every location within a single Voronoi cell is *closer*
    # to that cell's centroid than to any other centroid.
    #
    # The lines in the diagram represent the **decision boundaries**
    # of our vector quantizer.
    #
    # When a new 2D vector (a new point) arrives, we don't need
    # to calculate its distance to all 5 centroids. We just
    # need to see which Voronoi cell it falls into. The centroid
    # of that cell is its quantized value.
    
    try:
        # Calculate the Voronoi geometry
        vor = Voronoi(centroid)
        
        # Plot the Voronoi lines (the decision boundaries)
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False,
                        line_colors='k', line_styles='--', line_alpha=0.6)
    except Exception as e:
        print(f"Could not compute Voronoi diagram (this can happen "
              f"if points are collinear): {e}")

    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    ax.set_title("2D Vector Quantizer (k-means) and Voronoi Regions")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    plt.grid(True)
    
    # Save the figure
    output_filename = 'test_kmeans_voronoi_py.png'
    plt.savefig(output_filename)
    print(f"Saved plot to '{output_filename}'")
    
    plt.show()
    print("--- End of Exercise 2 ---")

if __name__ == '__main__':
    run_exercise_2()