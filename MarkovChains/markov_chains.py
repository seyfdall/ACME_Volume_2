# markov_chains.py
"""Volume 2: Markov Chains.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la


class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        (fill this out)
    """
    # Problem 1
    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            # >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        # Raise value error if A is not square or is not column stochastic
        if A.shape[0] != A.shape[1]:
            raise ValueError("The matrix A is not square")
        if not np.allclose(A.sum(axis=0), np.ones(A.shape[1])):
            raise ValueError("The matrix A is not column stochastic")
        self.A = A

        # Construct states if needed
        if states is None:
            states = [i for i in range(len(A))]
        self.labels = states

        # Construct the dictionary
        self.d = {}
        for i in range(len(states)):
            self.d[states[i]] = i

    # Problem 2
    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        # Get the column of the transition matrix
        column = self.A[:, self.d[state]]

        # Draw from the categorical distribution
        next_column_index = np.argmax(np.random.multinomial(1, column))

        # Find label in dictionary
        for label in self.labels:
            if self.d[label] == next_column_index:
                return label

    # Problem 3
    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        walk = [start]
        next = start

        # Cycle through N times running transition() method
        for i in range(N - 1):
            next = self.transition(next)
            walk.append(next)

        return walk

    # Problem 3
    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        next = start
        path = [start]

        # Cycle until we get the stop state
        while next != stop:
            next = self.transition(next)
            path.append(next)

        return path

    # Problem 4
    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """

        # Generate state distribution vector
        x = np.random.dirichlet(np.ones(len(self.A)), size=1)[0]

        # Iterate until steady state vector found
        for _ in range(maxiter):
            x_new = self.A @ x
            if la.norm(x_new - x) < tol:
                return x_new
            x = x_new

        # Steady state distribution vector not found - raise value error
        raise ValueError("There was no convergence within maxiter:", maxiter, "iterations.")


class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        (fill this out)
    """
    # Problem 5
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        unique_words = set()
        sentences = []

        with open(filename, 'r', encoding="utf8") as infile:
            s = infile.read().strip()
            sentences = s.split('\n')
            unique_words = set(s.split())

        # Build labels
        self.labels = []
        for word in unique_words:
            self.labels.append(word)
        self.labels.insert(0, "$tart")
        self.labels.append("$top")

        # Initialize matrix and build dictionary
        self.A = np.zeros((len(self.labels), len(self.labels)))
        self.d = {}
        for i in range(len(self.labels)):
            self.d[self.labels[i]] = i

        # Cycle through sentences building transition matrix
        for sentence in sentences:
            words = sentence.split()
            words.insert(0, "$tart")
            words.append("$top")
            for i in range(len(words) - 1):
                x = self.d[words[i]]
                y = self.d[words[i + 1]]
                self.A[y][x] += 1

        self.A[-1][-1] = 1

        # Normalize the new matrix
        col_sums = self.A.sum(axis=0, keepdims=True)
        self.A = self.A / col_sums
        print(self.A)

    # Problem 6
    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            # >>> yoda = SentenceGenerator("yoda.txt")
            # >>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """
        path = self.path("$tart", "$top")
        path.pop(0)
        path.pop()
        return ' '.join(path)


def test_prob_1():
    markov = MarkovChain(np.array([[.5, .8], [.5, .2]]), states=["A", "B"])
    markov = MarkovChain(np.array([[.5, .2]]), states=["A", "B"])


def test_prob_2():
    markov = MarkovChain(np.array([[.5, .8], [.5, .2]]), states=["A", "B"])
    print('\n')
    print(markov.transition('A'))
    markov = MarkovChain(np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]), states=["A", "B", "C", "D"])
    print('\n')
    next = markov.transition('A')
    print(next)
    next = markov.transition(next)
    print(next)
    next = markov.transition(next)
    print(next)
    next = markov.transition(next)
    print(next)


def test_prob_3():
    markov = MarkovChain(np.array([[.5, .8], [.5, .2]]), states=["A", "B"])
    print('\n')
    print(markov.walk('A', 3))
    print('\n')
    print(markov.path('A', 'B'))


def test_prob_4():
    markov = MarkovChain(np.array([[.5, .8], [.5, .2]]), states=["A", "B"])
    markov.steady_state()
    markov = MarkovChain(np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]), states=["A", "B", "C"])
    markov.steady_state()


def test_prob_6():
    yoda = SentenceGenerator("yoda.txt")
    print("\n")
    print(yoda.babble())

