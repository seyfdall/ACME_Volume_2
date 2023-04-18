# gradient_methods.py
"""Volume 2: Gradient Descent Methods.
<Name> Dallin Seyfried
<Class> 001
<Date> 2/23/2023
"""

import scipy.optimize as opt
import scipy.linalg as la
import numpy as np

# Problem 1
def steepest_descent(f, Df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    post = x0
    iters = 0
    converged = False

    # Run gradient descent till exceed maxiters or less than tolerance
    for _ in range(maxiter):
        iters += 1
        pre = post
        # Find the minimizer
        phi = lambda alpha: f(pre - alpha * Df(pre).T)
        alpha_k = opt.minimize_scalar(phi).x
        # Update Post
        post = pre - alpha_k * Df(pre).T
        # Check Tolerance
        if la.norm(Df(post), ord=np.inf) < tol:
            converged = True
            break

    return post, converged, iters


# Test Problem 1
def test_steepest_descent():
    f = lambda x: x[0]**4 + x[1]**4 + x[2]**4
    Df = lambda x: np.array([4*x[0]**3, 4*x[1]**3, 4*x[2]**3])
    x0 = np.array([1, 2, 3])
    print(steepest_descent(f, Df, x0))


# Problem 2
def conjugate_gradient(Q, b, x0, tol=1e-4):
    """Solve the linear system Qx = b with the conjugate gradient algorithm.

    Parameters:
        Q ((n,n) ndarray): A positive-definite square matrix.
        b ((n, ) ndarray): The right-hand side of the linear system.
        x0 ((n,) ndarray): An initial guess for the solution to Qx = b.
        tol (float): The convergence tolerance.

    Returns:
        ((n,) ndarray): The solution to the linear system Qx = b.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    r0 = Q @ x0 - b
    d0 = -r0
    k = 0
    n = len(b)

    # Conjugate Descent
    while la.norm(r0, ord=np.inf) >= tol and k < n:
        # Find minimizer with line search
        a0 = r0 @ r0 / (d0 @ Q @ d0)
        # Update x and other attributes of Gradient Descent
        x1 = x0 + a0 * d0
        r1 = r0 + a0 * Q @ d0
        b1 = r1 @ r1 / (r0 @ r0)
        d1 = -r1 + b1 * d0
        k = k + 1
        x0 = x1
        r0 = r1
        d0 = d1

    return x0, la.norm(r0, ord=np.inf) < tol, k


# Test Problem 2
def test_conjugate_gradient():
    n = 4
    A = np.random.random((n, n))
    Q = A.T @ A
    b, x0 = np.random.random((2,n))
    x = la.solve(Q, b)
    x_test = conjugate_gradient(Q, b, x0)
    print(np.allclose(Q @ x, b))
    print(np.allclose(Q @ x_test, b))


# Problem 3
def nonlinear_conjugate_gradient(f, df, x0, tol=1e-5, maxiter=10000):
    """Compute the minimizer of f using the nonlinear conjugate gradient
    algorithm.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    # Initialize beginning approximation
    r0 = -df(x0).T
    d0 = r0
    a0 = opt.minimize_scalar(lambda alpha: f(x0 + alpha * d0)).x
    x1 = x0 + a0 * d0
    k = 1

    # Cycle through non-linear conjugate gradient
    while la.norm(r0, ord=np.inf) >= tol and k < maxiter:
        r1 = -df(x1).T
        b1 = (r1 @ r1) / (r0 @ r0)
        d1 = r1 + b1 * d0
        a1 = opt.minimize_scalar(lambda alpha: f(x1 + alpha * d1)).x
        x1 = x1 + a1 * d1
        r0 = r1
        d0 = d1
        k = k + 1

    return x1, la.norm(r0, ord=np.inf) < tol, k


# Test Problem 3
def test_nonlinear_conjugate_gradient():
    # rosen = lambda x: (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    # rosen_der = lambda x: np.array([202*x[0] - 200*x[1] - 2,-200*x[0] + 200 * x[1]])
    print('\n')
    print(opt.fmin_cg(opt.rosen, np.array([2, 2]), fprime=opt.rosen_der))
    print(nonlinear_conjugate_gradient(opt.rosen, opt.rosen_der, np.array([2, 2])))

    def f(x):
        Q = np.array([
            [1, 2, 3, 4, 5],
            [2, 10, 6, 7, 8],
            [3, 6, 15, 9, 10],
            [4, 7, 9, 20, 11],
            [5, 8, 10, 11, 25]
        ])
        b = np.array([0, 0, 0, 0, 0])
        return x @ Q @ x / 2 - b @ x

    def df(x):
        Q = np.array([
            [1, 2, 3, 4, 5],
            [2, 10, 6, 7, 8],
            [3, 6, 15, 9, 10],
            [4, 7, 9, 20, 11],
            [5, 8, 10, 11, 25]
        ])
        b = np.array([0, 0, 0, 0, 0])
        return x @ Q - b.T

    print(opt.fmin_cg(f, np.array([2, 2, 2, 2, 2]), fprime=df))
    # print(nonlinear_conjugate_gradient(f, df, np.array([2, 2, 2, 2, 2])))


# Problem 4
def prob4(filename="linregression.txt",
          x0=np.array([-3482258, 15, 0, -2, -1, 0, 1829])):
    """Use conjugate_gradient() to solve the linear regression problem with
    the data from the given file, the given initial guess, and the default
    tolerance. Return the solution to the corresponding Normal Equations.
    """
    # Load the data and construct A and y
    data = np.loadtxt(filename)
    y = data[:,0]
    A = np.c_[np.ones(len(data)), data[:,1:]]

    # Build the matrix and vector for conjugate gradient
    Q = A.T @ A
    b = A.T @ y

    return conjugate_gradient(Q, b, x0)


# Test Problem 4
def test_prob4():
    print("\n")
    print(prob4())


# Problem 5
class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def fit(self, x, y, guess):
        """Choose the optimal beta values by minimizing the negative log
        likelihood function, given data and outcome labels.

        Parameters:
            x ((n,) ndarray): An array of n predictor variables.
            y ((n,) ndarray): An array of n outcome variables.
            guess (array): Initial guess for beta.
        """
        m = len(x)
        neg_log_likelihood = lambda b0, b1: np.sum(np.log(1 + np.exp(-(b0 + b1*x))) + (1 - y)*(b0 + b1*x))
        # test = opt.fmin_cg(neg_log_likelihood, guess)
        pass

    def predict(self, x):
        """Calculate the probability of an unlabeled predictor variable
        having an outcome of 1.

        Parameters:
            x (float): a predictor variable with an unknown label.
        """
        raise NotImplementedError("Problem 5 Incomplete")


# Test Problem 5
def test_logistic_regression1D():
    log_reg = LogisticRegression1D()
    x = np.array([1,2,3,4])
    y = np.array([0,0,1,1])
    log_reg.fit(x,y,[0,1])


# Problem 6
def prob6(filename="challenger.npy", guess=np.array([20., -1.])):
    """Return the probability of O-ring damage at 31 degrees Farenheit.
    Additionally, plot the logistic curve through the challenger data
    on the interval [30, 100].

    Parameters:
        filename (str): The file to perform logistic regression on.
                        Defaults to "challenger.npy"
        guess (array): The initial guess for beta.
                        Defaults to [20., -1.]
    """
    raise NotImplementedError("Problem 6 Incomplete")
