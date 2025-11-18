import numpy as np
import matplotlib.pyplot as plt


def visualize_dp(d, path, title):
    """
    Draws a colored matrix with values in each cell and overlays the DP path.

    Parameters:
        d    : distortion matrix (2D numpy array)
        path : list of (row, col) tuples representing the DP path
        title: plot title
    """
    M, N = d.shape

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title, fontsize=14)

    # Display matrix as colored squares
    cmap = plt.get_cmap("Blues")
    ax.imshow(d, cmap=cmap, origin="upper")

    # Show values inside squares
    for r in range(M):
        for c in range(N):
            ax.text(
                c, r,
                str(int(d[r, c])),
                ha="center",
                va="center",
                color="white" if d[r, c] > np.max(d) / 2 else "black",
                fontsize=12
            )

    # Plot the path
    pr = [p[0] for p in path]
    pc = [p[1] for p in path]
    ax.plot(pc, pr, marker="o", color="red", linewidth=2, markersize=8)

    # Invert y-axis so row 0 is at the top
    ax.invert_yaxis()

    ax.set_xlabel("Column index (k)", fontsize=12)
    ax.set_ylabel("Row index (j)", fontsize=12)
    ax.set_xticks(range(N))
    ax.set_yticks(range(M))
    ax.grid(color="black", linewidth=0.5)

    plt.tight_layout()
    plt.show()


def dp_example_results():
    """Returns the distortion matrix and the two paths from the earlier DP results."""
    d = np.array([
        [1, 1, 2, 2, 1],
        [1, 4, 3, 4, 2],
        [5, 2, 2, 1, 4],
        [3, 3, 3, 4, 5],
        [1, 4, 3, 4, 2]
    ], dtype=float)

    # From your computed output
    path_problem_1 = [(0, 0), (1, 0), (2, 1), (2, 2), (2, 3), (3, 4), (4, 4)]

    path_problem_2 = [(0, 0), (2, 1), (2, 2), (2, 3), (4, 4)]

    return d, path_problem_1, path_problem_2


if __name__ == "__main__":
    d, p1, p2 = dp_example_results()

    visualize_dp(d, p1, "DP Matching – Problem 1\nColored Matrix and Optimal Path")
    visualize_dp(d, p2, "DP Matching – Problem 2\nColored Matrix and Optimal Path")
