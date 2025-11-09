import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import correlate

def lpc_autocorr(x, order):
    """
    Computes the Linear Predictive Coding (LPC) coefficients using the
    autocorrelation method (Yule-Walker equations). This is a replacement
    for the deprecated scipy.signal.lpc and mimics the standard MATLAB lpc
    behavior in speech processing.

    :param x: The input signal frame.
    :param order: The LPC order (p).
    :return: (a, sigma_square)
             a: LPC coefficients [1, a1, a2, ..., ap] (numpy array)
             sigma_square: Residual variance (float)
    """

    # 1. Compute autocorrelation coefficients r[0] to r[p]
    # We use 'full' correlation to get the unbiased R[0] at the center.
    r_full = correlate(x, x, mode='full')

    # Find the index of lag 0 (the center)
    center_index = len(x) - 1

    # Extract R[0] to R[p] (indices center_index to center_index + order)
    r = r_full[center_index : center_index + order + 1]

    R0 = r[0]
    r_vector = r[1:] # r[1] to r[p] -> The 'r' vector in R*a = -r (Yule-Walker)

    # 2. Solve Yule-Walker equations R*a = -r
    # We use solve_toeplitz for the efficient Levinson-Durbin solution.
    try:
        # a_coeffs is the vector of coefficients [a1, a2, ..., ap]
        # For a symmetric Toeplitz matrix (autocorrelation), solve_toeplitz
        # only needs the first row 'c' and the right-hand side vector 'b'.
        # Explicitly cast check_finite to boolean to avoid ValueError
        a_coeffs = solve_toeplitz(r[:order], b=-r_vector, check_finite=True)
    except np.linalg.LinAlgError:
        # Fallback for singular matrix (e.g., all zeros input)
        a_coeffs = np.zeros(order)

    # 3. Compute residual variance (sigma_square)
    # sigma_square = R[0] + sum(a_k * R[k]) for k=1 to p
    # R[0] is r[0], the sum term is np.dot(a_coeffs, r_vector)
    sigma_square = R0 + np.dot(a_coeffs, r_vector)

    # 4. Construct the final coefficient vector [1, a1, a2, ..., ap]
    a = np.concatenate(([1.0], a_coeffs))

    return a, sigma_square