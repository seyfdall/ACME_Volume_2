# interior_point_linear.py
"""Volume 2: Interior Point for Linear Programs.
<Name> Dallin Seyfried
<Class> 001
<Date> 4/1/2023
"""

import numpy as np
from scipy import linalg as la
from scipy.stats import linregress
from matplotlib import pyplot as plt


# Auxiliary Functions ---------------------------------------------------------
def starting_point(A, b, c):
    """Calculate an initial guess to the solution of the linear program
    min c^T x, Ax = b, x>=0.
    Reference: Nocedal and Wright, p. 410.
    """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A @ A.T)
    x = A.T @ B @ b
    lam = B @ A @ c
    mu = c - (A.T @ lam)

    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    # Perturb x and mu so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    return x, lam, mu

# Use this linear program generator to test your interior point method.
def randomLP(j,k):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Parameters:
        j (int >= k): number of desired constraints.
        k (int): dimension of space in which to optimize.
    Returns:
        A ((j, j+k) ndarray): Constraint matrix.
        b ((j,) ndarray): Constraint vector.
        c ((j+k,), ndarray): Objective function with j trailing 0s.
        x ((k,) ndarray): The first 'k' terms of the solution to the LP.
    """
    A = np.random.random((j,k))*20 - 10
    A[A[:,-1]<0] *= -1
    x = np.random.random(k)*10
    b = np.zeros(j)
    b[:k] = A[:k,:] @ x
    b[k:] = A[k:,:] @ x + np.random.random(j-k)*10
    c = np.zeros(j+k)
    c[:k] = A[:k,:].sum(axis=0)/k
    A = np.hstack((A, np.eye(j)))
    return A, b, -c, x


# Problems --------------------------------------------------------------------
def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """Solve the linear program min c^T x, Ax = b, x>=0
    using an Interior Point method.

    Parameters:
        A ((m,n) ndarray): Equality constraint matrix with full row rank.
        b ((m, ) ndarray): Equality constraint vector.
        c ((n, ) ndarray): Linear objective function coefficients.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    n = len(c)
    m = len(b)

    # Define F
    def F(x, lam, mu):
        lag_f = np.hstack((A.T @ lam + mu - c, A @ x - b, np.diag(mu) @ x))
        return lag_f

    # Do a linear search to find next step
    def search_direction(x, lam, mu):
        # Find the derivative of F
        top_row = np.hstack((np.zeros((n, n)), A.T, np.eye(n)))
        mid_row = np.hstack((A, np.zeros((m, m)), np.zeros((m, n))))
        bot_row = np.hstack((np.diag(mu), np.zeros((n, m)), np.diag(x)))
        A_mat = np.vstack((top_row, mid_row, bot_row))

        # Calculate the right hand side
        v = x @ mu / n
        b_vec = -F(x, lam, mu) + np.hstack((np.zeros(n + m), np.ones(n) * v / 10))

        # lu_factor A_mat to get compatible with lu_solve
        lu, piv = la.lu_factor(A_mat)

        # lu_solve for deltas
        search_dir = la.lu_solve((lu, piv), b_vec)
        return search_dir

    # Calculate step size
    def step_size(x, mu, deltas):
        # Get specific x and mu deltas
        x_deltas = deltas[:n]
        lam_deltas = deltas[n:n+m]
        mu_deltas = deltas[n+m:]

        # Calculate maxes
        mu_mask = mu_deltas < 0
        x_mask = x_deltas < 0
        alpha_max = 1 if np.sum(mu_mask) == 0 else np.min(-mu[mu_mask] / mu_deltas[mu_mask])
        x_max = 1 if np.sum(x_mask) == 0 else np.min(-x[x_mask] / x_deltas[x_mask])

        # Calculate alpha and delta
        alpha = min(1, 0.95 * alpha_max)
        delta = min(1, 0.95 * x_max)
        return alpha, delta, [x_deltas, lam_deltas, mu_deltas]

    # Get the starting point
    x, lam, mu = starting_point(A, b, c)

    # Cycle niter times to get optimum and optimizer
    for i in range(niter):
        # Compute search direction
        direction = search_direction(x, lam, mu)

        # Compute step size
        alpha, delta, deltas = step_size(x, mu, direction)

        # Take the step
        x = x + delta * deltas[0]
        lam = lam + alpha * deltas[1]
        mu = mu + alpha * deltas[2]

        # Check if optimal
        if x @ mu / n < tol:
            break

    return x, c @ x


# Test interior point
def test_interior_point():
    j, k = 7, 5
    A, b, c, x = randomLP(j, k)
    point, value = interiorPoint(A, b, c)
    print('\n')
    print(x)
    print(point[:k])
    print(np.allclose(x, point[:k]))
    # A, b, c_neg, x = randomLP(2, 2)
    # opt, val = interiorPoint(A, b, -c_neg)


# Problem 5
def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""
    # Load data
    data = np.loadtxt('simdata.txt')
    m = data.shape[0]
    n = data.shape[1] - 1

    # Initialize vectors c, y, and x
    c = np.zeros(3 * m + 2 * (n + 1))
    c[:m] = 1
    y = np.empty(2*m)
    y[::2] = -data[:, 0]
    y[1::2] = data[:, 0]
    x = data[:, 1:]

    # Create the A matrix
    A = np.ones((2 * m, 3 * m + 2 * (n + 1)))
    A[::2, :m] = np.eye(m)
    A[1::2, :m] = np.eye(m)
    A[::2, m:m + n] = -x
    A[1::2, m:m + n] = x
    A[::2, m + n:m + 2 * n] = x
    A[1::2, m + n:m + 2 * n] = -x
    A[::2, m + 2 * n] = -1
    A[1::2, m + 2 * n + 1] = -1
    A[:, m + 2 * n + 2:] = -np.eye(2 * m, 2 * m)

    # Get the solution point
    sol = interiorPoint(A, y, c, niter=10)[0]

    # Get beta and the b vector
    beta = sol[m: m + n] - sol[m + n: m + 2 * n]
    b = sol[m + 2 * n] - sol[m + 2 * n + 1]

    # Plot least squares solution to compare results
    slope, intercept = linregress(data[:, 1], data[:, 0])[:2]
    domain = np.linspace(0, 10, 200)
    plt.plot(domain, domain * slope + intercept, label="Least Squares")
    plt.plot(domain, domain * beta + b, label="LAD")
    plt.scatter(data[:, 1], data[:, 0], label="Data Points")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Test leastAbsoluteDeviations
def test_LAD():
    leastAbsoluteDeviations()
