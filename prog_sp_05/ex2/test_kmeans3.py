# test_kmeans3.py
# (Translation of test_kmeans3.m)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d

def run_test_kmeans3():
    plt.close('all')
    print("--- Running test_kmeans3.py ---")

    # Generation of sampled vectors (2D, 4-group Gaussian distribution)
    X1 = np.random.randn(3000, 2) + 5 * np.ones((3000, 2))
    X2 = 2 * np.random.randn(3000, 2) + 0 * np.ones((3000, 2))
    X3 = 2 * np.random.randn(3000, 2) - 2 * np.ones((3000, 2))
    X4 = 3 * np.random.randn(3000, 2) - 5 * np.ones((3000, 2))
    X = np.concatenate((X1, X2, X3, X4), axis=0)

    n_clusters = 5
    print(f"Running k-means with {n_clusters} clusters...")
    
    # kmeans (generalized Lloyd)
    # n_init=1 mimics MATLAB's 'Replicates', 1
    kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=42)
    
    # 'idx' will contain cluster assignments (0 to 4)
    idx = kmeans.fit_predict(X) 
    # 'centroid' contains the cluster centers
    centroid = kmeans.cluster_centers_

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the clustered data points
    colors = ['r', 'b', 'g', 'y', 'c']
    for i in range(n_clusters):
        # Select points belonging to the current cluster
        cluster_points = X[idx == i]
        # Plot them with the specified color and marker
        ax.plot(cluster_points[:, 0], cluster_points[:, 1], '.', 
                color=colors[i], markersize=8, label=f'Cluster {i+1}')

    # Plot the centroids
    ax.plot(centroid[:, 0], centroid[:, 1], 'kx', 
            markersize=10, markeredgewidth=2, label='Centroids')
    ax.plot(centroid[:, 0], centroid[:, 1], 'ko', 
            markersize=10, markeredgewidth=2, markerfacecolor='none')
    
    # Voronoi Diagram
    # We plot this last, with no points, to overlay the lines
    try:
        vor = Voronoi(centroid)
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False,
                        line_colors='k', line_styles='--', line_alpha=0.6)
    except Exception as e:
        print(f"Could not compute Voronoi diagram (likely due to collinear points): {e}")

    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    
    # Save the figure
    output_filename = 'test_kmeans3_py.png'
    plt.savefig(output_filename)
    print(f"Saved plot to '{output_filename}'")

    # Show the plot
    plt.title("k-means Clustering and Voronoi Diagram (Python)")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    run_test_kmeans3()