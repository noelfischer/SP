# prog_sp_03-2.py test_filter
import numpy as np
from scipy import signal

# --- Custom filter function equivalent ---
def my_filter(b, a, x):
    """
    Custom implementation of a Discrete-Time Linear Time-Invariant (LTI) filter.

    The difference equation is:
    a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[q]*x[n-q]
              - a[1]*y[n-1] - a[2]*y[n-2] - ... - a[p]*y[n-p]

    Since the MATLAB 'filter' assumes a[0] = 1, we implement the direct form II
    based on the equation:
    y[n] = (b[0]*x[n] + ... + b[q]*x[n-q]) - (a[1]*y[n-1] + ... + a[p]*y[n-p])

    Parameters:
    b (numpy.array): Numerator coefficients [b0, b1, ..., bq]
    a (numpy.array): Denominator coefficients [a0, a1, ..., ap] (where a0=1)
    x (numpy.array): Input signal [x[0], x[1], ..., x[N-1]]

    Returns:
    numpy.array: Output signal y
    """

    # nq1 = length(b); # b[1],...,b[q+1]: b0,b1,...,bq (nq = q+1). In Python, length of b.
    nq1 = len(b)
    # np1 = length(a); # a[1],...,a[p+1]; a0=1, a1,a2,...,ap (np = p+1). In Python, length of a.
    np1 = len(a)
    # nx = length(x); # x[1],,,,,x[nx]. In Python, length of x.
    nx = len(x)

    print('this') # Equivalent to display('this') in MATLAB

    # Initialize output array (MATLAB uses 1-based indexing for length, Python uses 0-based)
    y = np.zeros(nx)

    # Loop through each output sample n (0 to nx-1 in Python)
    for n in range(nx):
        current_sum = 0

        # FIR part: Sum of current and past inputs (x terms)
        # MATLAB: for j = 1 : nq1. Python: for j in range(nq1) (j goes from 0 to nq1-1)
        for j in range(nq1):
            # MATLAB index: n-j+1 (since MATLAB uses 1-based indexing for n and j)
            # Python index: n-j

            # Check for non-negative index (equivalent to (n-j+1) >= 1 in MATLAB)
            input_index = n - j
            if input_index >= 0:
                # MATLAB: sum = sum + b(j)*x(n-j+1);
                # Python: current_sum += b[j] * x[input_index]
                current_sum += b[j] * x[input_index]

        # IIR part: Sum of past outputs (y terms)
        # MATLAB: for i = 2: np1. Python: for i in range(1, np1) (i goes from 1 to np1-1)
        # Note: i=1 in Python corresponds to the a[1] coefficient (second element)
        for i in range(1, np1):
            # MATLAB index: n-i+1 (since MATLAB uses 1-based indexing for n and i)
            # Python index: n-i

            # Check for non-negative index (equivalent to (n-i+1) >= 1 in MATLAB)
            output_index = n - i
            if output_index >= 0:
                # MATLAB: sum = sum - a(i)*y(n-i+1);
                # Python: current_sum -= a[i] * y[output_index]
                current_sum -= a[i] * y[output_index]

        # Store the calculated output sample
        # MATLAB: y(n) = sum;
        # Python: y[n] = current_sum
        y[n] = current_sum

    return y

# Main script equivalent
def main():
    # Clear variables and close figures are not needed in Python script context
    # unless working in an interactive environment like an IDE or Jupyter notebook.

    # Define filter coefficients (numerator)
    b = np.array([1, 2]) # Filter coefficients (numerator, v in original comment)
    # Define filter coefficients (denominator)
    a = np.array([1, 0, 0])

    # Input signal u
    x = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float) # Convert to float

    # Calculate output using standard library function (w in original comment)
    # Equivalent to MATLAB's filter(b, a, x)
    y_lib = signal.lfilter(b, a, x)

    # Calculate output using the custom filter function
    y_custom = my_filter(b, a, x)

    # Display results (using print for Python)
    print("Output from scipy.signal.lfilter (y_lib):")
    print(y_lib)
    print("\nOutput from custom implementation (y_custom):")
    print(y_custom)

# Execute the main function when the script is run
if __name__ == '__main__':
    main()

