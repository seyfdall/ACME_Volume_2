# oneD_optimization.py
"""Volume 2: One-Dimensional Optimization.
<Name> Dallin Seyfried
<Class> 001
<Date> 2/2/2023
"""

import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt
from scipy.optimize import linesearch
from jax import numpy as jnp
from jax import grad

# Problem 1
def golden_section(f, a, b, tol=1e-5, maxiter=100):
    """Use the golden section search to minimize the unimodal function f.

    Parameters:
        f (function): A unimodal, scalar-valued function on [a,b].
        a (float): Left bound of the domain.
        b (float): Right bound of the domain.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    # Set initial approx and golden ratio
    x_0 = x_1 = (a + b) / 2
    g_r = (1 + np.sqrt(5)) / 2
    i = 0
    converged = False

    # Cycle through maxiter times at most to find approximation
    for _ in range(maxiter):
        i += 1

        c = (b - a) / g_r
        a_approx = b - c
        b_approx = a + c

        # Update approximations
        if f(a_approx) <= f(b_approx):
            b = b_approx
        else:
            a = a_approx
        x_1 = (a + b) / 2

        # Break if found minimizer
        if abs(x_0 - x_1) < tol:
            converged = True
            break
        x_0 = x_1

    return x_1, converged, i


# Test problem 1
def test_prob_1():
    # Set up f and domain
    f = lambda x: np.exp(x) - 4 * x
    domain = np.linspace(0, 3, 100)

    # Approximate using both custom and scipy functions
    min_approx_1 = golden_section(f, 0, 3)[0]
    min_approx_2 = opt.golden(f, brack=(0, 3), tol=.001)

    if abs(min_approx_1 - min_approx_2) > .01:
        print("Approximations are not equal")

    # Plotting the points
    plt.plot(domain, f(domain), label="f(x)")
    plt.scatter(min_approx_1, f(min_approx_1), label="golden_section", color="g")
    plt.scatter(min_approx_2, f(min_approx_2), label="scipy golden", color="r")
    plt.title("Golden Ratio Approximation")
    plt.tight_layout()
    plt.legend()
    plt.show()


# Problem 2
def newton1d(df, d2f, x0, tol=1e-5, maxiter=100):
    """Use Newton's method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        d2f (function): The second derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    post = x0
    converged = False

    # Cycle up to maxiter times and return if converged
    i = 0
    for _ in range(maxiter):
        i += 1
        pre = post
        # Including alpha for backtracking
        post = pre - df(pre) / d2f(pre)
        if np.linalg.norm(post - pre) < tol:
            converged = True
            break


    return post, converged, i


# Testing problem 2 - newton1d
def test_prob_2():
    # Set up functions and its derivatives
    f = lambda x: x**2 + np.sin(5*x)
    df = lambda x: 2*x + 5*np.cos(5*x)
    d2f = lambda x: 2 - 25*np.sin(5*x)
    x0 = 0

    # Compute approximations
    min_approx_1 = newton1d(df, d2f, x0)[0]
    min_approx_2 = opt.newton(df, x0=x0, fprime=d2f, tol=1e-10, maxiter=500)

    # Compare approximations
    assert abs(min_approx_1 - min_approx_2) < .01


# Problem 3
def secant1d(df, x0, x1, tol=1e-5, maxiter=100):
    """Use the secant method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        x1 (float): Another guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    # Set up initial values
    x2 = x1
    converged = False
    i = 0

    # Cycle maxiter times to compute approximation
    for _ in range(maxiter):
        i += 1
        # Compute derivatives at x1 and x0 once
        f_prime_x1 = df(x1)
        f_prime_x0 = df(x0)

        # Find x_k+1
        numerator = x0 * f_prime_x1 - x1 * f_prime_x0
        denominator = f_prime_x1 - f_prime_x0
        x2 = numerator / denominator

        # Check convergence
        if abs(x2 - x1) < tol:
            converged = True
            break

        x0 = x1
        x1 = x2

    return x2, converged, i


def test_prob_3():
    # Set up function and derivative
    f = lambda x: x**2 + np.sin(x) + np.sin(10*x)
    df = lambda x: 2*x + np.cos(x) + 10*np.cos(10*x)
    domain = np.linspace(-np.pi, np.pi, 100)
    x0, x1 = 0, -1

    # Run approximations
    min_approx_1 = secant1d(df, x0, x1)[0]
    min_approx_2 = opt.newton(df, x0=0, tol=1e-10, maxiter=500)

    # Plotting the points
    plt.plot(domain, f(domain), label="f(x)")
    plt.scatter(min_approx_1, f(min_approx_1), label="secant1d", color="r")
    plt.scatter(min_approx_2, f(min_approx_2), label="opt.newton", color="g")
    plt.title("Secant Optimization")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Problem 4
def backtracking(f, Df, x, p, alpha=1, rho=.9, c=1e-4):
    """Implement the backtracking line search to find a step size that
    satisfies the Armijo condition.

    Parameters:
        f (function): A function f:R^n->R.
        Df (function): The first derivative (gradient) of f.
        x (float): The current approximation to the minimizer.
        p (float): The current search direction.
        alpha (float): A large initial step length.
        rho (float): Parameter in (0, 1).
        c (float): Parameter in (0, 1).

    Returns:
        alpha (float): Optimal step size.
    """
    # Compute and store Dfp and fx once
    Dfp = Df(x) @ p
    fx = f(x)

    # Cycle to find a suitable aplpha
    while f(x + alpha * p) > fx + c*alpha*Dfp:
        alpha = rho*alpha

    return alpha


def test_prob_4():
    # Set up functions
    f = lambda x: x[0]**2 + x[1]**2 + x[2]**2
    Df = lambda x: np.array([2*x[0], 2*x[1], 2*x[2]])

    # Compute approximations of alpha - step size
    x = jnp.array([150., .03, 40.])
    p = jnp.array([-.5, -100., -4.5])
    phi = lambda alpha: f(x + alpha * p)
    dphi = grad(phi)
    alpha1, _ = linesearch.scalar_search_armijo(phi, phi(0.), dphi(0.))
    alpha2 = backtracking(f, Df, x, p)
    print(f"armijo alpha = {alpha1}")
    print(f"backtracking alpha = {alpha2}")