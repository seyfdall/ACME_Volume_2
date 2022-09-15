# numpy_intro.py
"""Python Essentials: Intro to NumPy.
<Name> Dallin Seyfried
<Class> Math 321 002
<Date> 09/01/22
"""
import numpy as np


def prob1():
    """ Define the matrices A and B as arrays. Return the matrix product AB. """
    # Initialize A and B numpy matrices
    A = np.array([[3, -1, 4], [1, 5, -9]])
    B = np.array([[2, 6, -5, 3], [5, -8, 9, 7], [9, -3, -2, -3]])
    # Return their matrix product
    return np.dot(A, B)


def prob2():
    """ Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A. """
    # Initialize array
    A = np.array([[3, 1, 4], [1, 5, 9], [-5, 3, 1]])
    # Calculate first term in formula
    ret_mat = -np.dot(A, np.dot(A, A))
    # Calculate and add second term in formula to return matrix
    ret_mat += 9 * np.dot(A, A)
    # Calculate and add third term in formula to return matrix
    ret_mat += -15 * A
    return ret_mat


def prob3():
    """ Define the matrices A and B as arrays using the functions presented in
    this section of the manual (not np.array()). Calculate the matrix product ABA,
    change its data type to np.int64, and return it.
    """
    # Initialize matrices
    A = np.triu(np.ones((7, 7)))
    B = -np.tril(np.ones((7, 7))) + 5 * np.triu(np.ones((7, 7))) - np.diag([5,5,5,5,5,5,5])

    # Calculate matrix product
    ret_mat = A @ B @ A

    # Change data type to np.int64 and return the new matrix
    typed_ret_mat = ret_mat.astype(np.int64)
    return typed_ret_mat


def prob4(A):
    """ Make a copy of 'A' and use fancy indexing to set all negative entries of
    the copy to 0. Return the resulting array.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    # Make a copy of the input matrix
    A_copy = np.copy(A)
    # Create a mask for the copy matrix
    mask = A_copy < 0
    # Run the mask on the matrix to change all negative values to zero
    A_copy[mask] = 0
    return A_copy

def prob5():
    """ Define the matrices A, B, and C as arrays. Use NumPy's stacking functions
    to create and return the block matrix:
                                | 0 A^T I |
                                | A  0  0 |
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    # Initializing A,B, and C numpy arrays
    A = np.array([[0, 2, 4], [1, 3, 5]])
    B = np.array([[3, 0, 0], [3, 3, 0], [3, 3, 3]])
    C = np.array([[-2, 0, 0], [0, -2, 0], [0, 0, -2]])

    # Initializing Identity and Zero matrices
    I = np.eye(3)
    Zeroes_1 = np.zeros((3, 3))
    Zeroes_2 = np.zeros((2, 2))
    Zeroes_3 = np.zeros((3, 2))
    Zeroes_4 = np.zeros((2, 3))

    # Creating each block column in the array using vstack
    col_1 = np.vstack((Zeroes_1, A, B))
    col_2 = np.vstack((np.transpose(A), Zeroes_2, Zeroes_3))
    col_3 = np.vstack((I, Zeroes_4, C))

    # Combining each column block into the whole array usiing hstack
    return np.hstack((col_1, col_2, col_3))


def prob6(A):
    """ Divide each row of 'A' by the row sum and return the resulting array.
    Use array broadcasting and the axis argument instead of a loop.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    # Calculating row sums into a 2-d array
    row_sums = A.sum(axis=1).reshape((len(A), 1))
    # Using array broadcasting to return the stochastic array
    return A / row_sums

def prob7():
    """ Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid. Use slicing, as specified in the manual.
    """
    # Load the grid
    grid = np.load("grid.npy")

    # Calculate the horizontal, vertical, and each diagonal maxes for the grid 2-d matrix
    max_horizontal = np.max(grid[:, :-3] * grid[:, 1:-2] * grid[:, 2:-1] * grid[:, 3:])
    max_vertical = np.max(grid[:-3, :] * grid[1:-2, :] * grid[2:-1, :] * grid[3:, :])
    max_diagonal_up_left = np.max(grid[:-3, :-3] * grid[1:-2, 1:-2] * grid[2:-1, 2:-1] * grid[3:, 3:])
    max_diagonal_up_right = np.max(grid[:- 3, 3:] * grid[1:-2, 2:-1] * grid[2:-1, 1:-2] * grid[3:, :-3])

    # Return the max of the maxes
    return max(max_horizontal, max_vertical, max_diagonal_up_left, max_diagonal_up_right)


if __name__ == "__main__":
    print(prob1())
    print(prob2())
    print(prob3())
    print(prob4(np.array([-3, -1, 3])))
    print(prob5())
    print(prob6(np.array([[1,1,0],[0,1,0],[1,1,1]])))
    print(prob7())

