"""Volume 2: Simplex

<Name> Dallin Seyfried
<Date> 03/07/2023
<Class> 001
"""

import numpy as np


# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        minimize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    # Problem 1
    def __init__(self, c, A, b):
        """Check for feasibility and initialize the dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        # Setup initial length values
        self.n = len(c)
        self.m = len(b)

        # Test origin feasibility
        zeros_arr = np.zeros(self.n)
        if np.min(A @ zeros_arr) > np.min(b):
            raise ValueError(f"Given system is infeasible at the origin.")

        # Set parameters as class attributes if feasible
        self.c = c
        self.A = A
        self.b = b

        # Generate dictionary
        self.dictionary = self._generatedictionary(self.c, self.A, self.b)

    # Problem 2
    def _generatedictionary(self, c, A, b):
        """Generate the initial dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.
        """

        m, n = A.shape
        dictionary = np.zeros((1 + m, 1 + m + n))
        # Slice in each piece.
        dictionary[0, 1:] = np.hstack((c, np.zeros(m)))
        dictionary[1:, 0] = b
        dictionary[1:, 1:] = np.hstack((-A, -np.eye(m)))
        return dictionary

    # Problem 3a
    def _pivot_col(self):
        """Return the column index of the next pivot column.
        """
        # Find the column index
        col_ind = np.where(self.dictionary[0, 1:] < 0)[0]
        return col_ind[0] + 1

    # Problem 3b
    def _pivot_row(self, index):
        """Determine the row index of the next pivot row using the ratio test
        (Bland's Rule).
        """
        # Calculate which rows are valid - if none then problem is unbounded
        neg_rows = np.insert(self.dictionary[1:, index] < 0, 0, False)
        if len(neg_rows) == 0:
            raise ValueError("Problem is unbounded")

        # Find the appropriate row_index/indices
        row_ind = np.argmin(np.abs(self.dictionary[neg_rows, 0] / self.dictionary[neg_rows, index]))

        # Blanch's rule to return first indexed
        if type(row_ind) is np.ndarray:
            row_ind = row_ind[0]

        return np.where(neg_rows != 0)[0][row_ind]

    # Problem 4
    def pivot(self):
        """Select the column and row to pivot on. Reduce the column to a
        negative elementary vector.
        """
        # Find where pivot is in array
        col_index = int(self._pivot_col())
        row_index = self._pivot_row(col_index)

        # Divide pivot row by abs(pivot)
        self.dictionary[row_index] /= -float(self.dictionary[row_index, col_index])

        # Simplify matrix by zeroing out pivot column with row operations
        for i in range(len(self.dictionary)):
            if i != row_index:
                self.dictionary[i] += self.dictionary[row_index] * self.dictionary[i, col_index]
                # self.dictionary[i] = self.dictionary[i] - self.dictionary[i, col_index] / self.dictionary[row_index, col_index] * self.dictionary[row_index, :]

    # Problem 5
    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The minimum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        # Pivot and cycle through the dictionary until solved or unbounded error found
        while True in (self.dictionary[0, 1:] < 0):
            self.pivot()

        # Calculate min
        min = self.dictionary[0, 0]
        independent = dict()
        dependent = dict()

        # Create dictionaries for independent and dependent variables
        for i in range(1, len(self.dictionary[0])):
            if self.dictionary[0, i] == 0:
                dependent[i - 1] = (self.dictionary[:, i] == -1) @ (self.dictionary[:, 0])
            else:
                independent[i - 1] = 0

        return tuple((min, dependent, independent))


# Test Simplex Class
def test_simplex_solver():
    # Test Simplex Solver Class
    # Initialize objective function and constraints.
    c = np.array([-3., -2.])
    b = np.array([2., 5, 7])
    A = np.array([[1., -1], [3, 1], [4, 3]])
    # Instantiate the simplex solver, then solve the problem.
    solver = SimplexSolver(c, A, b)
    sol = solver.solve()
    print(sol)


# Problem 6
def prob6(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        ((n,) ndarray): the number of units that should be produced for each product.
    """
    data = np.load(filename)
    A = data.f.A
    p = data.f.p
    m = data.f.m
    d = data.f.d

    # Construct A
    A = np.row_stack((A, np.eye(len(d))))

    # Construct b
    b = np.concatenate((m, d))

    # Construct c
    c = p * -1

    # Solve
    simple_solver = SimplexSolver(c, A, b)
    min, dep, ind = simple_solver.solve()
    return np.array([dep[0], dep[1], dep[2], dep[3]])


# Test Problem 6
def test_prob6():
    print("\n")
    print(prob6())
