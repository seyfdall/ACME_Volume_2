# polynomial_interpolation.py
"""Volume 2: Polynomial Interpolation.
<Name> Dallin Seyfried
<Class> 001
<Date> 1/19/2023
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator
import numpy.linalg as la


# Problems 1 and 2
def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """
    n = len(xint)
    m = len(points)

    # Compute the denominator
    L_j_denom = np.zeros(len(xint))
    for j in range(len(xint)):
        prod = 1
        for k in range(len(xint)):
            if j != k:
                prod *= (xint[j] - xint[k])
        L_j_denom[j] = prod

    # Construct the nxm matrix
    L_j = np.zeros((n,m))
    for i in range(m):
        for j in range(n):
            prod = 1
            for k in range(n):
                if j != k:
                    prod *= (points[i] - xint[k])
            L_j[j][i] = prod / L_j_denom[j]

    # Define the interpolating polynomial y values by multiplying L_j by yint and summing
    for i in range(n):
        L_j[i] *= yint[i]
    y_points = np.sum(L_j, axis=0)
    return y_points


# Problems 3 and 4
class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        self.x = xint
        self.y = yint
        self.n = len(self.x)

        # Calcluate weights
        n = len(self.x)
        w = np.ones(n)

        # Calculate the capacity of the interval
        C = (np.max(xint) - np.min(xint) / 4)
        shuffle = np.random.permutation(n - 1)
        for j in range(n):
            temp = (xint[j] - np.delete(xint, j)) / C
            temp = temp[shuffle]
            w[j] /= np.product(temp)

        self.w = w

    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """

        def _eval_poly(x0):
            " takes in x (array), y (array), x0 (float), returns interpolation at x (float)"
            # If the value already exists in the array-
            if x0 in self.x:
                return self.y[list(self.x).index(x0)]

            # Set weights
            weights = self.w

            # Calculate the fraction
            weight_frac = weights / (x0 - self.x)

            # Calculate the value
            val = np.sum(weight_frac * self.y) / np.sum(weight_frac)
            return val

        return np.array([_eval_poly(point) for point in points])

    # Problem 4
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        # Calculate and add one weight at a time
        for i, x_i in enumerate(xint):
            new_weight = 1 / np.product([x_i - x_k] for x_k in self.x)

            # Update weights
            for j in range(len(self.w)):
                self.w[j] /= (self.x[j] - x_i)

            # Append x value and sort the list
            self.x = np.append(self.x, x_i)
            self.x = np.sort(self.x)

            # Find the added values' indices
            sorter = np.argsort(self.x)
            value_index = sorter[np.searchsorted(self.x, x_i, sorter=sorter)]

            # For each index, insert corresponding y and w elements
            self.y = np.insert(self.y, value_index, yint[i])
            self.w = np.insert(self.w, value_index, new_weight)


# Problem 5
def prob5():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    domain = np.linspace(-1, 1, 400)
    f_point = lambda x: 1 / (1 + 25 * x**2)
    f = np.vectorize(f_point)

    n_list = [2**i for i in range(2, 9)]
    equal_error = []
    cheby_error = []

    for n in n_list:
        points = np.linspace(-1, 1, n)
        cheby_points = [np.cos(j*np.pi/n) for j in range(n+1)]
        actual = f(domain)

        # Get Barycentric interpolation using equally spaced points
        poly_equal = BarycentricInterpolator(points)
        poly_equal.set_yi(f(points))
        equal = poly_equal(domain)
        equal_error.append(la.norm(actual - equal, ord=np.inf))

        # Get Barycentric interpolation using cheby points
        poly_cheby = BarycentricInterpolator(cheby_points)
        poly_cheby.set_yi(f(cheby_points))
        cheby = poly_cheby(domain)
        cheby_error.append(la.norm(actual - cheby, ord=np.inf))

    # Plot the errors
    plt.loglog(n_list, equal_error, label="Equally Spaced")
    plt.loglog(n_list, cheby_error, label="Cheby Spaced")
    plt.ylabel("Error")
    plt.xlabel("n")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Problem 6
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    # Vectorize passed in function
    f_vectorized = np.vectorize(f)

    # Generate chebyshev extremizer points
    cheby_points = [np.cos(j * np.pi / n) for j in range(n+1)]

    # Create and extend vector of points
    f_vec = np.array(f_vectorized(cheby_points))
    f_vec = np.append(f_vec, f_vec[1:n][::-1])

    # Calculate DFT and scale appropriately
    coeffs = np.real(np.fft.fft(f_vec))[:n] * 2 * 1 / (2 * n)
    coeffs[0] /= 2
    coeffs[-1] /= 2
    return coeffs


# Problem 7
def prob7(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    # Setup to interpolate at extrema chebyshev points
    fx = lambda a, b, n: .5*(a+b + (b-a) * np.cos(np.arange(n+1) * np.pi / n))
    a, b = 0, 366 - 1/24
    data = np.load('airdata.npy')
    domain = np.linspace(0, b, 8784)
    points = fx(a, b, n)
    temp = np.abs(points - domain.reshape(8784, 1))
    temp2 = np.argmin(temp, axis=0)
    poly = Barycentric(domain[temp2], data[temp2])

    # Plot the original data
    plt.subplot(211)
    plt.plot(domain, data)
    plt.title("original data")

    # Plot the approximating polynomial
    plt.subplot(212)
    plt.plot(domain, poly(domain))
    plt.title("approximating polynomial")
    plt.tight_layout()
    plt.show()


# prob5()
# domain = np.linspace(-1, 1, 100)
# sample = np.linspace(-1, 1, 5)
# test = Barycentric(sample, 1 / (1 + 25*sample**2))
# plt.plot(domain, test(sample))
# Define f(x) = -3 + 2x^2 - x^3 + x^4 by its (ascending) coefficients.
# f = lambda x: -3 + 2*x**2 - x**3 + x**4
# pcoeffs = [-3, 0, 2, -1, 1]
# ccoeffs = np.polynomial.chebyshev.poly2cheb(pcoeffs)
# custom = chebyshev_coeffs(f, 5)
# # The following callable objects are equivalent to f().
# fpoly = np.polynomial.Polynomial(pcoeffs)
# fcheb = np.polynomial.Chebyshev(ccoeffs)
# prob7(200)
