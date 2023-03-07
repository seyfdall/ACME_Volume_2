"""Volume 2: Simplex

<Name>
<Date>
<Class>
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
        self._generatedictionary(self.c, self.A, self.b)

    # Problem 2
    def _generatedictionary(self, c, A, b):
        """Generate the initial dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.
        """
        # Create components of the D matrix
        A_bar = np.column_stack((A, np.eye(self.m)))
        c_bar = np.append(c, np.zeros(self.m))

        # Construct the D matrix
        D_top = np.insert(c_bar, 0, 0)
        D_bottom = np.column_stack((b, -A_bar))
        D = np.row_stack((D_top, D_bottom))

        self.D = D


    # Problem 3a
    def _pivot_col(self):
        """Return the column index of the next pivot column.
        """
        # Find the column index
        col_ind = np.where(self.D[0, 1:] < 0)[0]
        return col_ind[0] + 1

    # Problem 3b
    def _pivot_row(self, index):
        """Determine the row index of the next pivot row using the ratio test
        (Bland's Rule).
        """
        # Calculate which rows are valid - if none then problem is unbounded
        neg_rows = np.insert(self.D[1:, index] < 0, 0, False)
        if len(neg_rows) == 0:
            raise ValueError("Problem is unbounded")

        # Find the appropriate row_index/indices
        row_ind = np.argmin(np.abs(self.D[neg_rows, 0] / self.D[neg_rows, index]))

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
        col_index = self._pivot_col()
        row_index = self._pivot_row(col_index)

        # Divide pivot row by abs(pivot)
        self.D[row_index, :] = self.D[row_index, :] / np.abs(self.D[row_index, col_index])

        # Simplify matrix by zeroing out pivot column with row operations
        for i in range(len(self.D)):
            if i != row_index:
                self.D[i, :] = self.D[i, :] - self.D[i, col_index] / self.D[row_index, col_index] * self.D[row_index, :]

    # Problem 5
    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The minimum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        # Pivot and cycle through the dictionary until solved or unbounded error found
        while True in (self.D[0, 1:] < 0):
            self.pivot()

        # Calculate min
        min = self.D[0, 0]
        independent = dict()
        dependent = dict()

        # Create dictionaries for independent and dependent variables
        for i in range(1, len(self.D[0])):
            if self.D[0, i] == 0:
                dependent[i - 1] = (self.D[:, i] == -1) @ (self.D[:, 0])
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
    resource_coeffs = data.f.A
    unit_prices = data.f.p
    available_resource_units = data.f.m
    demand_constraints = data.f.d

    # Construct A
    A = np.row_stack((resource_coeffs, np.eye(len(demand_constraints))))

    # Construct b
    b = np.concatenate((available_resource_units, demand_constraints))

    # Construct c
    c = unit_prices * -1

    # Solve
    simple_solver = SimplexSolver(c, A, b)
    min, dep, ind = simple_solver.solve()
    return np.array([int(dep[0]), int(dep[1]), int(dep[2]), int(dep[3])])


# Test Problem 6
def test_prob6():
    print(prob6())
