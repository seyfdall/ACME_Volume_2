# dynamic_programming.py
"""Volume 2: Dynamic Programming.
<Name> Dallin Seyfried
<Class> 001
<Date> 04/28/2023
"""

import numpy as np
import matplotlib.pyplot as plt


def calc_stopping(N):
    """Calculate the optimal stopping time and expected value for the
    marriage problem.

    Parameters:
        N (int): The number of candidates.

    Returns:
        (float): The maximum expected value of choosing the best candidate.
        (int): The index of the maximum expected value.
    """
    # Define an array containing the values of each candidate
    c_vals = [0] * N
    maximum = 0
    index = N

    # Cycle backwards calculating each value
    for i in range(N - 1)[::-1]:
        # Fill in the value being careful with indices
        i_max = max((i + 1) / (i + 2) * c_vals[i + 1] + 1 / N, c_vals[i + 1])
        c_vals[i] = i_max

        # If i_max is greater than maximum update maximum and index
        if i_max > maximum:
            maximum = i_max
            index = i

    return maximum, index + 1


# Test problem 1
def test_calc_stopping():
    print(calc_stopping(10))


# Problem 2
def graph_stopping_times(M):
    """Graph the optimal stopping percentage of candidates to interview and
    the maximum probability against M.

    Parameters:
        M (int): The maximum number of candidates.

    Returns:
        (float): The optimal stopping percent of candidates for M.
    """
    # Define domain and ranges
    percentages = [0] * M
    probabilities = [0] * M
    domain = range(M)

    # Calculate percentages and probabilities
    for N in range(M):
        value, stopping_point = calc_stopping(N + 1)
        percentages[N] = stopping_point / (N + 1)
        probabilities[N] = value

    # Graph the results
    plt.plot(domain[2:], percentages[2:], label="Percentages")
    plt.plot(domain[2:], probabilities[2:], label="Probabilities")
    plt.title("Percentages vs Maximum probabilities over N")
    plt.tight_layout()
    plt.legend()
    plt.show()

    return percentages[-1]


# Test Problem 2
def test_graph():
    print('\n')
    print(graph_stopping_times(1000))


# Problem 3
def get_consumption(N, u=lambda x: np.sqrt(x)):
    """Create the consumption matrix for the given parameters.

    Parameters:
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        u (function): Utility function.

    Returns:
        C ((N+1,N+1) ndarray): The consumption matrix.
    """
    # Construct the partition vector w
    w = np.array([n / N for n in range(N + 1)])

    # Construct the consumption matrix for the given parameters
    C = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            # If column index greater than row set to zero
            if i <= j:
                C[i,j] = 0
            else:
                C[i,j] = u(w[i] - w[j])

    return C


# Define test for problem 3
def test_get_consumption():
    C = get_consumption(N=4, u=lambda x: x)
    print('\n')
    print(C)


# Problems 4-6
def eat_cake(T, N, B, u=lambda x: np.sqrt(x)):
    """Create the value and policy matrices for the given parameters.

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        A ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            value of having w_i cake at time j.
        P ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            number of pieces to consume given i pieces at time j.
    """
    # Set Matrices to zeros to start
    A = np.zeros((N+1, T+1))
    P = np.zeros((N+1, T+1))

    # Construct the partition vector w
    w = np.array([n / N for n in range(N + 1)])

    # Calculate the last column of A and P for problem 4 and 6
    A[:, T] = u(w)
    P[:, T] = w

    # Construct A by constructing CV^t for each time for problem 5
    for t in range(T)[::-1]:

        # Define function to build the CV matrix at time t
        def cv_func(i,j):
            return u(w[i] - w[j]) + B * A[j, t + 1]

        # Build the CV_t matrix and convert nans to zero
        CV_t = np.nan_to_num(np.fromfunction(cv_func, shape=(N+1, N+1), dtype=int))

        # Define Ait and Pit from CV_t
        for i in range(N+1):
            A[i,t] = np.max(CV_t[i,:])
            j = np.argmax(CV_t[i,:])
            if type(j) is np.ndarray:
                j = j[0]
            P[i,t] = w[i] - w[j]

    return A, P


# Test problem 4-6
def test_eat_cake():
    eat_cake(T=3, N=4, B=0.9, u=lambda x: np.sqrt(x))


# Problem 7
def find_policy(T, N, B, u=np.sqrt):
    """Find the most optimal route to take assuming that we start with all of
    the pieces. Show a graph of the optimal policy using graph_policy().

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        ((T,) ndarray): The matrix describing the optimal percentage to
            consume at each time.
    """
    # Get our policy and value matrices
    A, P = eat_cake(T, N, B, u)

    # Calculate policy vector from P
    policy = [0] * (T+1)
    n = N
    for i in range(T+1):
        policy[i] = P[n, i]
        n = n - round(P[n, i] * N)

    return policy


# Test problem 7
def test_find_policy():
    print(find_policy(T=3, N=4, B=0.9, u=lambda x: np.sqrt(x)))