# nearest_neighbor.py
"""Volume 2: Nearest Neighbor Search.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from scipy.spatial import KDTree
from scipy.stats import mode
from matplotlib import pyplot as plt


# Problem 1
def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    # Array broadcast to find reduced matrix then use axis=0 with argmin to find best solution
    reduced_X = X - z
    index = np.argmin(la.norm(reduced_X, axis=1))
    return X[index], la.norm(X[index])


# Problem 2: Write a KDTNode class.
class KDTNode:
    """Node class for K-D Trees.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        value ((k,) ndarray): a coordinate in k-dimensional space.
        pivot (int): the dimension of the value to make comparisons on.
    """

    def __init__(self, x):

        # Check type and throw error if not match
        if type(x) is not np.ndarray:
            raise TypeError(str(x), "constructor input is not of type np.ndarray")

        # Initialize starting values
        self.value = x
        self.left = None
        self.right = None
        self.pivot = None


# Problems 3 and 4
class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
            ValueError: if data is already in the tree
        """

        # If the root is empty
        if self.root is None:
            new_node = KDTNode(data)
            new_node.pivot = 0
            self.root = new_node
            self.k = len(data)

        # If the tree is nonempty
        else:
            def _step(current, new_node):
                if np.allclose(current.value, new_node.value):
                    raise ValueError(new_node.value, "already exists in the tree")
                elif current.value[current.pivot] >= new_node.value[current.pivot]:
                    if current.left is None:
                        current.left = new_node
                        new_node.pivot = (current.pivot + 1) % self.k
                    else:
                        _step(current.left, new_node)
                else:
                    if current.right is None:
                        current.right = new_node
                        new_node.pivot = (current.pivot + 1) % self.k
                    else:
                        _step(current.right, new_node)

            _step(self.root, KDTNode(data))

    # Problem 4
    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        # Defining inner recursive function to search
        def _kd_search(current, nearest, d_star):
            # If current is none then return our best solution
            if current is None:
                return nearest, d_star

            # Initialize values
            x = current.value
            i = current.pivot
            d = la.norm(x - z)

            # Compare new and old distances and update if better
            if d < d_star:
                nearest = current
                d_star = d

            # Search to the left
            if z[i] < x[i]:
                nearest, d_star = _kd_search(current.left, nearest, d_star)
                # Search to the right if needed
                if z[i] + d_star >= x[i]:
                    nearest, d_star = _kd_search(current.right, nearest, d_star)
            # Search to the right
            else:
                nearest, d_star = _kd_search(current.right, nearest, d_star)
                #Search to the left if needed
                if z[i] - d_star <= x[i]:
                    nearest, d_star = _kd_search(current.right, nearest, d_star)
            return nearest, d_star

        node, d_star = _kd_search(self.root, self.root, la.norm(self.root.value - z))
        return node.value, d_star

    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    """A k-nearest neighbors classifier that uses SciPy's KDTree to solve
    the nearest neighbor problem efficiently.
    """

    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.tree = None
        self.labels = None

    def fit(self, X, y):
        """Accept m x k Numpy array X and a 1-dimensional Numpy array y with
        m entries.  As in problems 1 and 4, each of the m rows
        of X represents a point in R^k. Here yi is the label corresponding to row i of X.

        Load a SciPy KDTree with the data in X. Save the tree and the labels as attributes.
        """
        self.tree = KDTree(X)
        self.labels = y

    def predict(self, z):
        """accept a 1-dimensional NumPy array z with k entries. Query the KDTree for
        the n_neighbors elements of X that are nearest to z and return the most common label
        of those neighbors. If there is a tie for the most common label (such as if k = 2 in Figure
        3.5), choose the alphanumerically smallest label.
        """
        distances, indices = self.tree.query(z, k=self.n_neighbors)
        return mode([self.labels[i] for i in indices])[0]

# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    # .91
    # Import the data and setup training and test variables
    data = np.load(filename)
    X_train = data["X_train"].astype(np.float)
    y_train = data["y_train"]
    X_test = data["X_test"].astype(np.float)
    y_test = data["y_test"]

    # Visualize one of the images
    plt.imshow(X_test[0].reshape((28, 28)), cmap="gray")
    plt.show()

    # Load a classifier from problem 5
    classifier = KNeighborsClassifier(n_neighbors)
    classifier.fit(X_train, y_train)

    # Compute classification accuracy
    correct = 0
    for i in range(len(X_test)):
        result = classifier.predict(X_test[i])
        if result == y_test[i]:
            correct += 1

    return correct / len(y_test)


def test_prob_1():
    """Testing problem 1 - exhaustive search"""
    A = np.random.random((6, 3))
    z = np.random.random(3)
    print(exhaustive_search(A, z))


def test_prob_2_3():
    """Testing problem 2 & 3 - constructor and """
    tree = KDT()
    tree.insert(np.array([3, 1, 4]))
    tree.insert(np.array([1, 2, 7]))
    tree.insert(np.array([4, 3, 5]))
    tree.insert(np.array([2, 0, 3]))
    tree.insert(np.array([2, 4, 5]))
    tree.insert(np.array([6, 1, 4]))
    tree.insert(np.array([1, 4, 3]))
    tree.insert(np.array([0, 5, 7]))
    tree.insert(np.array([5, 2, 5]))
    print(str(tree))


def test_prob_4():
    """Testing problem 4 - constructor and """
    data = np.random.random((100, 5))
    target = np.random.random(5)
    tree = KDTree(data)
    min_distance, index = tree.query(target)
    print("\n")
    print("Scipy min_distance:", min_distance)
    print("Scipy tree indexed value:", tree.data[index])

    my_tree = KDT()
    for datum in data:
        my_tree.insert(datum)

    min_distance, index = tree.query(target)
    print("KDT min_distance:", min_distance)
    print("KDT tree indexed value:", tree.data[index])


def test_prob_6():
    print(prob6(4))

