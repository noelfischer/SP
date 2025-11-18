import numpy as np
from typing import List, Tuple, Optional

INF = 1e12


def map_to_user_coords(d_array: np.ndarray) -> np.ndarray:
    """Convert numpy array where row 0 is top into user coords where m=0 is bottom."""
    M, N = d_array.shape
    d_user = np.empty_like(d_array)
    for m in range(M):
        for n in range(N):
            d_user[m, n] = d_array[M - 1 - m, n]
    return d_user


def map_user_to_array_index(m: int, n: int, M: int) -> Tuple[int, int]:
    """Map user coords (m,n) to numpy array indices (row, col). row = M-1-m, col = n"""
    return M - 1 - m, n


def traceback_path(bp: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Trace back path from goal to start using backpointer bp. bp contains predecessor tuples or None."""
    path: List[Tuple[int, int]] = []
    cur: Optional[Tuple[int, int]] = goal
    while cur is not None:
        path.append(cur)
        cur = bp[cur]
    path.reverse()
    return path


def dp_problem_1_usercoords(d_user: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Problem 1 in user coords.
    Predecessors of (m,n):
      (m-1, n)         weight 1
      (m-1, n-1)       weight 2
      (m,   n-1)       weight 1
    Path restriction: |m - n| <= 2
    """
    M, N = d_user.shape
    g = np.full((M, N), INF)
    bp: np.ndarray = np.full((M, N), None, dtype=object)

    g[0, 0] = d_user[0, 0]

    for n in range(N):           # left->right
        for m in range(M):       # bottom->top
            if m == 0 and n == 0:
                continue
            if abs(m - n) > 2:
                continue

            candidates = []
            # (m-1, n)
            if m - 1 >= 0:
                candidates.append((g[m - 1, n] + d_user[m, n], (m - 1, n)))
            # (m-1, n-1) weight 2
            if m - 1 >= 0 and n - 1 >= 0:
                candidates.append((g[m - 1, n - 1] + 2 * d_user[m, n], (m - 1, n - 1)))
            # (m, n-1)
            if n - 1 >= 0:
                candidates.append((g[m, n - 1] + d_user[m, n], (m, n - 1)))

            if candidates:
                best_cost, best_prev = min(candidates, key=lambda x: x[0])
                g[m, n] = best_cost
                bp[m, n] = best_prev

    path = traceback_path(bp, (0, 0), (M - 1, N - 1))
    return g, path


def dp_problem_2_usercoords(d_user: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Problem 2 in user coords.
    Allowed outgoing moves from (m,n):
      (m,   n+1)         right
      (m+1, n+1)         diag up-right
      (m+2, n+1)         diag two-up one-right

    Therefore predecessors of (m,n) are:
      (m,   n-1)
      (m-1, n-1)
      (m-2, n-1)

    All weights = 1 (per the exercise)
    Path restriction: |m - n| <= 2
    """
    M, N = d_user.shape
    g = np.full((M, N), INF)
    bp: np.ndarray = np.full((M, N), None, dtype=object)

    g[0, 0] = d_user[0, 0]

    for n in range(N):           # left->right
        for m in range(M):       # bottom->top
            if m == 0 and n == 0:
                continue
            if abs(m - n) > 2:
                continue

            candidates = []
            # predecessor (m, n-1)
            if n - 1 >= 0:
                candidates.append((g[m, n - 1] + d_user[m, n], (m, n - 1)))
            # predecessor (m-1, n-1)
            if m - 1 >= 0 and n - 1 >= 0:
                candidates.append((g[m - 1, n - 1] + d_user[m, n], (m - 1, n - 1)))
            # predecessor (m-2, n-1)
            if m - 2 >= 0 and n - 1 >= 0:
                candidates.append((g[m - 2, n - 1] + d_user[m, n], (m - 2, n - 1)))

            if candidates:
                best_cost, best_prev = min(candidates, key=lambda x: x[0])
                g[m, n] = best_cost
                bp[m, n] = best_prev

    path = traceback_path(bp, (0, 0), (M - 1, N - 1))
    return g, path


def pretty_print_matrix_with_origin_bottom_left(mat: np.ndarray, title: str):
    M, N = mat.shape
    print(title)
    print(f"(Printed top row first. user m=0 is bottom row. shape={M}x{N})")
    for m in range(M - 1, -1, -1):
        row = ["{:.2f}".format(x) if isinstance(x, float) else str(x) for x in mat[m]]
        print("  " + "  ".join(row))
    print()


def path_to_moves(path: List[Tuple[int, int]]) -> List[str]:
    """Convert user coord path into textual moves from one node to the next."""
    moves = []
    for (m0, n0), (m1, n1) in zip(path, path[1:]):
        dm, dn = m1 - m0, n1 - n0
        if dm == 0 and dn == 1:
            moves.append("right")
        elif dm == 1 and dn == 1:
            moves.append("diag_up_right")
        elif dm == 2 and dn == 1:
            moves.append("diag_2up_1right")
        elif dm == 1 and dn == 0:
            moves.append("up")
        else:
            moves.append(f"move(dm={dm},dn={dn})")
    return moves


def print_path_with_array_indices(path_user: List[Tuple[int, int]], M: int):
    print("Path in user coords (m bottom->top, n left->right):")
    print(path_user)
    arr_idx = [map_user_to_array_index(m, n, M) for (m, n) in path_user]
    print("Path as array indices (row, col) for the original numpy array (row 0 is top):")
    print(arr_idx)
    print("Moves along path:")
    print(path_to_moves(path_user))
    print()


if __name__ == "__main__":
    # original array as given (row 0 is the first list provided)
    d_array = np.array([
        [1, 1, 2, 2, 1],
        [1, 4, 3, 4, 2],
        [5, 2, 2, 1, 4],
        [3, 3, 3, 4, 5],
        [1, 4, 3, 4, 2]
    ], dtype=float)

    M, N = d_array.shape

    print("\n=== DP Matching Exercise 7 (user coordinates) ===\n")
    print("Original numpy array (row 0 is top):")
    for row in d_array:
        print(" ", row)
    print()

    d_user = map_to_user_coords(d_array)
    pretty_print_matrix_with_origin_bottom_left(d_user, "Distortion matrix d (user coords)")

    # Problem 1
    g1, path1 = dp_problem_1_usercoords(d_user)
    pretty_print_matrix_with_origin_bottom_left(g1, "Cumulative cost g for Problem 1 (user coords)")
    print("Minimum total distortion (Problem 1) at goal (m=M-1,n=N-1):", g1[-1, -1])
    print_path_with_array_indices(path1, M)

    # Problem 2
    g2, path2 = dp_problem_2_usercoords(d_user)
    pretty_print_matrix_with_origin_bottom_left(g2, "Cumulative cost g for Problem 2 (user coords)")
    print("Minimum total distortion (Problem 2) at goal (m=M-1,n=N-1):", g2[-1, -1])
    print_path_with_array_indices(path2, M)

    print("Done.")
