# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
<Name> Dallin Seyfried
<Class> 001
<Date> 3/22/2023
"""

import cvxpy as cp
import numpy as np


def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # Initialize objective function
    x = cp.Variable(3)
    c = np.array([2, 1, 3])
    objective = cp.Minimize(c.T @ x)

    # Construct constraints
    A = np.array([[1, 2, 0]])
    B = np.array([0, 1, -4])
    C = np.array([-2, -10, -3])
    P = np.eye(3)
    constraints = [A @ x <= 3, B @ x <= 1, C @ x <= -12, P @ x >= 0]

    # Assemble problem and then solve it
    problem = cp.Problem(objective, constraints)
    opt_val = problem.solve()
    optimizer = x.value

    return optimizer, opt_val


# Test prob1
def test_prob1():
    optimizer, opt_val = prob1()
    print('\n')
    print(optimizer)
    print(opt_val)


# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    n = len(A[0])
    # Initialize objective function
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm(x, 1))

    # Construct constraints
    constraints = [A @ x == b]

    # Assemble problem then solve it
    problem = cp.Problem(objective, constraints)
    opt_val = problem.solve()
    optimizer = x.value

    return optimizer, opt_val


# Test Problem 2
def test_prob_2():
    A = np.array([
        [1, 2, 1, 1],
        [0, 3, -2, -1]
    ])
    b = np.array([7,4])
    optimizer, opt_val = l1Min(A, b)
    print('\n')
    print(optimizer)
    print(opt_val)


# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # Initialize objective function
    p = cp.Variable(6)
    c = np.array([4, 7, 6, 8, 8, 9])
    objective = cp.Minimize(c.T @ p)

    # Construct constraints
    P = np.eye(6)
    supply_1 = np.array([1, 1, 0, 0, 0, 0])
    supply_2 = np.array([0, 0, 1, 1, 0, 0])
    supply_3 = np.array([0, 0, 0, 0, 1, 1])
    demand_1 = np.array([1, 0, 1, 0, 1, 0])
    demand_2 = np.array([0, 1, 0, 1, 0, 1])
    constraints = [
        P @ p >= 0,
        supply_1 @ p == 7,
        supply_2 @ p == 2,
        supply_3 @ p == 4,
        demand_1 @ p == 5,
        demand_2 @ p <= 8
    ]

    # Assemble problem and then solve it
    problem = cp.Problem(objective, constraints)
    opt_val = problem.solve()
    optimizer = p.value

    return optimizer, opt_val


# Test prob3
def test_prob3():
    optimizer, opt_val = prob3()
    print('\n')
    print(optimizer)
    print(opt_val)


# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # Define Q matrix
    Q = np.array([
        [3, 2, 1],
        [2, 4, 2],
        [1, 2, 3]
    ])

    # Define r array
    r = np.array([3, 0, 1])

    # Define cp var x
    x = cp.Variable(3)

    # Define the problem and return values
    problem = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, Q) + r.T @ x))
    opt_val = problem.solve()
    optimizer = x.value

    return optimizer, opt_val


# Test prob4
def test_prob4():
    optimizer, opt_val = prob4()
    print('\n')
    print(optimizer)
    print(opt_val)


# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # Initialize objective function
    n = len(A[0])
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm(A @ x - b, 2))

    # Initialize constraints
    P = np.eye(n)
    constraints = [cp.sum(x) == 1, P @ x >= 0]

    # Assemble problem and then solve it
    problem = cp.Problem(objective, constraints)
    opt_val = problem.solve()
    optimizer = x.value

    return optimizer, opt_val


# Test Problem 5
def test_prob_5():
    A = np.array([
        [1, 2, 1, 1],
        [0, 3, -2, -1]
    ])
    b = np.array([7,4])
    optimizer, opt_val = prob5(A, b)
    print('\n')
    print(optimizer)
    print(opt_val)


# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """	 
    data = np.load("food.npy", allow_pickle=True)
    n = 18

    # Initialize Objective function
    x = cp.Variable(n)
    p = data[:, 0]
    objective = cp.Minimize(p.T @ x)

    # Determine constraints
    s = data[:, 1]
    P = np.eye(n)
    constraints = [
        (s * data[:, 2]).T @ x <= 2000,
        (s * data[:, 3]).T @ x <= 65,
        (s * data[:, 4]).T @ x <= 50,
        (s * data[:, 5]).T @ x >= 1000,
        (s * data[:, 6]).T @ x >= 25,
        (s * data[:, 7]).T @ x >= 46,
        P @ x >= 0
    ]

    # Assemble problem and then solve it
    problem = cp.Problem(objective, constraints)
    opt_val = problem.solve()
    optimizer = x.value

    return optimizer, opt_val


# Test prob6
def test_prob6():
    optimizer, opt_val = prob6()
    print('\n')
    print(optimizer)
    print(opt_val)