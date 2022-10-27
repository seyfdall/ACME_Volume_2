# breadth_first_search.py
"""Volume 2: Breadth-First Search.
<Name>
<Class>
<Date>
"""

from collections import deque
import networkx as nx
from matplotlib import pyplot as plt


# Problems 1-3
class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors.

    Attributes:
        d (dict): the adjacency dictionary of the graph.
    """
    def __init__(self, adjacency={}):
        """Store the adjacency dictionary as a class attribute"""
        self.d = dict(adjacency)

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        return str(self.d)

    # Problem 1
    def add_node(self, n):
        """Add n to the graph (with no initial edges) if it is not already
        present.

        Parameters:
            n: the label for the new node.
        """
        if n not in self.d:
            self.d[n] = set()

    # Problem 1
    def add_edge(self, u, v):
        """Add an edge between node u and node v. Also add u and v to the graph
        if they are not already present.

        Parameters:
            u: a node label.
            v: a node label.
        """
        if u not in self.d:
            self.d[u] = set()
        if v not in self.d:
            self.d[v] = set()
        self.d[u].add(v)
        self.d[v].add(u)

    # Problem 1
    def remove_node(self, n):
        """Remove n from the graph, including all edges adjacent to it.

        Parameters:
            n: the label for the node to remove.

        Raises:
            KeyError: if n is not in the graph.
        """
        if n not in self.d:
            raise KeyError(str(n) + " node not in the graph")
        self.d[n] = None

        # Cycle through the nodes in the graph and remove all other edges from other
        # nodes connecting to n
        for node in self.d:
            if n in node:
                node.remove(n)

    # Problem 1
    def remove_edge(self, u, v):
        """Remove the edge between nodes u and v.

        Parameters:
            u: a node label.
            v: a node label.

        Raises:
            KeyError: if u or v are not in the graph, or if there is no
                edge between u and v.
        """
        if self.d[u] is None or self.d[v] is None or self.d[u][v] is None or self.d[v][u] is None:
            raise KeyError("remove_edge() node or edge is not in the graph")
        self.d[u].remove(v)
        self.d[v].remove(u)

    # Problem 2
    def traverse(self, source):
        """Traverse the graph with a breadth-first search until all nodes
        have been visited. Return the list of nodes in the order that they
        were visited.

        Parameters:
            source: the node to start the search at.

        Returns:
            (list): the nodes in order of visitation.

        Raises:
            KeyError: if the source node is not in the graph.
        """
        # Check if source is in graph
        if source not in self.d:
            raise KeyError(str(source) + " is not in the graph")

        # Initialize data structures
        V = []
        Q = deque()
        M = set()
        Q.append(source)
        M.add(source)

        # Cycle through the graph performing a breadth first search
        while len(Q) > 0:
            current = Q.popleft()
            V.append(current)
            for node in self.d[current]:
                if node not in M:
                    Q.append(node)
                    M.add(node)

        return V

    # Problem 3
    def shortest_path(self, source, target):
        """Begin a BFS at the source node and proceed until the target is
        found. Return a list containing the nodes in the shortest path from
        the source to the target, including endoints.

        Parameters:
            source: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from source to target,
                including the endpoints.

        Raises:
            KeyError: if the source or target nodes are not in the graph.
        """
        # If source or target not in graph raise error
        if source not in self.d:
            raise KeyError(str(source) + " source not in graph")
        if source not in self.d:
            raise KeyError(str(target) + " target not in graph")

        # Initialize data structures
        V = []
        Q = deque()
        M = set()
        P = {}
        Q.append(source)
        M.add(source)
        found_target = False

        # Cycle through the graph performing a breadth first search
        while len(Q) > 0 and found_target is False:
            current = Q.popleft()
            for node in self.d[current]:
                if node not in M:
                    # Add key-value pair mapping visited node to visiting node
                    # If the node is the target, break. Otherwise, continue BFS
                    P[node] = current
                    if node == target:
                        found_target = True
                        break
                    Q.append(node)
                    M.add(node)

        # Cycle back through the dictionary to generate V from target to source
        node = target
        V.append(node)
        while node in P:
            node = P[node]
            V.append(node)

        # Return the reverse of V to get source to target
        return V[::-1]


# Problems 4-6
class MovieGraph:
    """Class for solving the Kevin Bacon problem with movie data from IMDb."""

    # Problem 4
    def __init__(self, filename="movie_data.txt"):
        """Initialize a set for movie titles, a set for actor names, and an
        empty NetworkX Graph, and store them as attributes. Read the speficied
        file line by line, adding the title to the set of movies and the cast
        members to the set of actors. Add an edge to the graph between the
        movie and each cast member.

        Each line of the file represents one movie: the title is listed first,
        then the cast members, with entries separated by a '/' character.
        For example, the line for 'The Dark Knight (2008)' starts with

        The Dark Knight (2008)/Christian Bale/Heath Ledger/Aaron Eckhart/...

        Any '/' characters in movie titles have been replaced with the
        vertical pipe character | (for example, Frost|Nixon (2008)).
        """
        self.movies = set()
        self.actors = set()
        self.G = nx.Graph()

        with open(filename, 'r', encoding="utf8") as infile:
            for line in infile.readlines():
                # Split each input line by '/' and build the nx Graph
                arr_line = line.strip().split('/')
                movie = arr_line.pop(0)
                self.movies.add(movie)
                for actor in arr_line:
                    self.actors.add(actor)
                    self.G.add_edge(movie, actor)

    # Problem 5
    def path_to_actor(self, source, target):
        """Compute the shortest path from source to target and the degrees of
        separation between source and target.

        Returns:
            (list): a shortest path from source to target, including endpoints and movies.
            (int): the number of steps from source to target, excluding movies.
        """
        path = nx.shortest_path(self.G, source, target)
        path_length = nx.shortest_path_length(self.G, source, target) // 2

        return path, path_length

    # Problem 6
    def average_number(self, target):
        """Calculate the shortest path lengths of every actor to the target
        (not including movies). Plot the distribution of path lengths and
        return the average path length.

        Returns:
            (float): the average path length from actor to target.
        """
        path_lengths = nx.shortest_path_length(self.G, target)
        average_length = 0
        p_l = []

        # Cycle through every actor adding up their shortest path lengths
        for actor in self.actors:
            average_length += path_lengths[actor] // 2
            p_l.append(path_lengths[actor] // 2)

        # Histogram plot of average number
        plt.hist(p_l, bins=[i-.5 for i in range(8)])
        plt.xlabel("Degree of Separation")
        plt.ylabel("Number of Actors")
        plt.title("Average_Number Histogram")
        plt.show()

        return average_length / len(self.actors)


def test_bfs():
    # Test BFS
    g = Graph()
    g.add_node('A')
    g.add_node('B')
    g.add_node('C')
    g.add_node('D')
    g.add_edge('A', 'D')
    g.add_edge('A', 'B')
    g.add_edge('C', 'D')
    g.add_edge('B', 'D')
    print(g.traverse('A'))
    print(g.shortest_path('A', 'C'))


def test_kevin_bacon():
    # Test Kevin bacon problem
    movies = MovieGraph()
    print(movies.average_number("Christopher Lee"))
