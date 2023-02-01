# quassian_quadrature.py
"""Volume 2: Gaussian Quadrature.
<Name> Dallin Seyfried
<Class> 001
<Date> 1/26/2023
"""
import numpy as np
import numpy.linalg as la
from scipy.integrate import quad, nquad
from scipy.stats import norm
from matplotlib import pyplot as plt


class GaussianQuadrature:
    """Class for integrating functions on arbitrary intervals using Gaussian
    quadrature with the Legendre polynomials or the Chebyshev polynomials.
    """
    # Problems 1 and 3
    def __init__(self, n, polytype="legendre"):
        """Calculate and store the n points and weights corresponding to the
        specified class of orthogonal polynomial (Problem 3). Also store the
        inverse weight function w(x)^{-1} = 1 / w(x).

        Parameters:
            n (int): Number of points and weights to use in the quadrature.
            polytype (string): The class of orthogonal polynomials to use in
                the quadrature. Must be either 'legendre' or 'chebyshev'.

        Raises:
            ValueError: if polytype is not 'legendre' or 'chebyshev'.
        """
        # Raise error if polytype doesn't match
        if polytype != "legendre" and polytype != "chebyshev":
            raise ValueError(f"polytype argument is not legendre or chebyshev")
        self.polytype = polytype

        self.n = n
        self.points_weights(self.n)

        # Define the reciprocal function
        if self.polytype == "legendre":
            self.reciprocal = lambda x: 1
        else:
            self.reciprocal = lambda x: np.sqrt(1-x**2)

    # Problem 2
    def points_weights(self, n):
        """Calculate the n points and weights for Gaussian quadrature.

        Parameters:
            n (int): The number of desired points and weights.

        Returns:
            points ((n,) ndarray): The sampling points for the quadrature.
            weights ((n,) ndarray): The weights corresponding to the points.
        """

        # Construct Legendre
        if self.polytype == "legendre":
            b_k = np.sqrt(np.array([k**2 / (4*k**2 - 1) for k in range(1, n)]))
            weight_value = 2

        # Construct Chebyshev
        else:
            b_k = np.sqrt(np.ones(n - 1) * (1/4))
            b_k[0] = np.sqrt(0.5)
            weight_value = np.pi

        # Create the Jacobi then calculate weights and return points and weights
        jacobi = np.diag(b_k, 1) + np.diag(b_k, -1)
        eig_vals, eig_vecs = la.eig(jacobi)
        weights = weight_value * np.array([eig_vecs[0][i] ** 2 for i in range(n)])

        self.points = eig_vals
        self.weights = weights

        return eig_vals, weights

    # Problem 3
    def basic(self, f):
        """Approximate the integral of a f on the interval [-1,1]."""
        g = f(self.points) * self.reciprocal(self.points)
        return np.dot(self.weights, g)

    # Problem 4
    def integrate(self, f, a, b):
        """Approximate the integral of a function on the interval [a,b].

        Parameters:
            f (function): Callable function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.

        Returns:
            (float): Approximate value of the integral.
        """
        # Use basic() to get approximate integral of f on interval [-1,1]
        # Then multiply by (b - a) / 2 to get a new approximation
        return (b - a) / 2 * self.basic(lambda x: f((b - a) / 2 * x + (a + b) / 2))

    # Problem 6.
    def integrate2d(self, f, a1, b1, a2, b2):
        """Approximate the integral of the two-dimensional function f on
        the interval [a1,b1]x[a2,b2].

        Parameters:
            f (function): A function to integrate that takes two parameters.
            a1 (float): Lower bound of integration in the x-dimension.
            b1 (float): Upper bound of integration in the x-dimension.
            a2 (float): Lower bound of integration in the y-dimension.
            b2 (float): Upper bound of integration in the y-dimension.

        Returns:
            (float): Approximate value of the integral.
        """
        coeff = (b1 - a1) * (b2 - a2) / 4
        sums = self.integrate(f, a1, b1) * self.integrate(f, a2, b2)
        return coeff * sums


# test = GaussianQuadrature(5, "chebyshev")
# prob1 = test.points_weights(5)
# f = lambda x, y: np.sin(x) + np.cos(y)
# print(nquad(f, [[-10, 10], [-1, 1]])[0])
# print(test.integrate2d(f, -10, 10, -1, 1))

# Problem 5
def prob5():
    """Use scipy.stats to calculate the "exact" value F of the integral of
    f(x) = (1/sqrt(2 pi))e^((-x^2)/2) from -3 to 2. Then repeat the following
    experiment for n = 5, 10, 15, ..., 50.
        1. Use the GaussianQuadrature class with the Legendre polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
        2. Use the GaussianQuadrature class with the Chebyshev polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
    Plot the errors against the number of points and weights n, using a log
    scale for the y-axis. Finally, plot a horizontal line showing the error of
    scipy.integrate.quad() (which doesn’t depend on n).
    """
    # Fix F and declare f
    f = lambda x: (1 / np.sqrt(2 * np.pi)) * np.exp((-x**2) / 2)
    a = -3
    b = 2
    F = norm.cdf(b) - norm.cdf(a)
    legendre_error = []
    chebyshev_error = []
    domain = [5*i for i in range(1, 11)]

    for n in domain:
        # Approximate F using Legendre polynomials
        leg = GaussianQuadrature(n, "legendre")
        leg_value = leg.integrate(f, a, b)
        legendre_error.append(abs(F - leg_value))

        # Approximate F using Chebyshev polynomials
        cheb = GaussianQuadrature(n, "chebyshev")
        cheb_value = cheb.integrate(f, a, b)
        chebyshev_error.append(abs(F - cheb_value))

    # Plot the errors
    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel("Error")
    plt.plot(domain, legendre_error, label="Legendre Error")
    plt.plot(domain, chebyshev_error, label="Chebyshev Error")
    plt.plot(domain, [abs(F - quad(f, a, b)[0])]*10, label="Scipy Quad")
    plt.legend()
    plt.title("Problem 5")
    plt.tight_layout()
    plt.show()





