# linked_lists.py
"""Volume 2: Linked Lists.
<Name>
<Class>
<Date>
"""


# Problem 1
import pytest


class Node:
    """A basic node class for storing data."""
    def __init__(self, data):
        """Store the data in the value attribute.
                
        Raises:
            TypeError: if data is not of type int, float, or str.
        """
        if type(data) != int and type(data) != float and type(data) != str:
            raise TypeError("Tell me why, data ain't type int, float, or str... I want it that way")
        else:
            self.value = data


class LinkedListNode(Node):
    """A node class for doubly linked lists. Inherits from the Node class.
    Contains references to the next and previous nodes in the linked list.
    """
    def __init__(self, data):
        """Store the data in the value attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        Node.__init__(self, data)       # Use inheritance to set self.value.
        self.next = None                # Reference to the next node.
        self.prev = None                # Reference to the previous node.

    def __str__(self):
        """str method to be used with __str__ method in LinkedList"""
        return repr(self.value)


# Problems 2-5
class LinkedList:
    """Doubly linked list data structure class.

    Attributes:
        head (LinkedListNode): the first node in the list.
        tail (LinkedListNode): the last node in the list.
    """
    def __init__(self):
        """Initialize the head and tail attributes by setting
        them to None, since the list is empty initially.
        """
        self.head = None
        self.tail = None
        self.size = 0

    def append(self, data):
        """Append a new node containing the data to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
        else:
            # If the list is not empty, place new_node after the tail.
            self.tail.next = new_node               # tail --> new_node
            new_node.prev = self.tail               # tail <-- new_node
            # Now the last node in the list is new_node, so reassign the tail.
            self.tail = new_node

        self.size += 1

    # Problem 2
    def find(self, data):
        """Return the first node in the list containing the data.

        Raises:
            ValueError: if the list does not contain the data.

        Examples:
            l = LinkedList()
            for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            node = l.find('b')
            node.value
            'b'
            l.find('f')
            ValueError: <message>
        """

        node = self.head
        while node is not None and node.value != data:
            node = node.next

        if node is None:
            raise ValueError("find() - Oh deary me, I'm afraid no such node exists")
        else:
            return node

    # Problem 2
    def get(self, i):
        """Return the i-th node in the list.

        Raises:
            IndexError: if i is negative or greater than or equal to the
                current number of nodes.

        Examples:
            l = LinkedList()
            for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            node = l.get(3)
            node.value
            'd'
            l.get(5)
            IndexError: <message>
        """
        if i < 0 or i > self.size - 1:
            raise IndexError("get() - i appears to be negative or greater than the size of the list")
        else:
            node = self.head
            while node is not None and i > 0:
                node = node.next
                i -= 1
            return node

    # Problem 3
    def __len__(self):
        """Return the number of nodes in the list.

        Examples:
            l = LinkedList()
            for i in (1, 3, 5):
            ...     l.append(i)
            ...
            len(l)
            3
            l.append(7)
            len(l)
            4
        """
        return self.size

    # Problem 3
    def __str__(self):
        """String representation: the same as a standard Python list.

        Examples:
            l1 = LinkedList()       |   >>> l2 = LinkedList()
            for i in [1,3,5]:       |   >>> for i in ['a','b',"c"]:
            ...     l1.append(i)        |   ...     l2.append(i)
            ...                         |   ...
            print(l1)               |   >>> print(l2)
            [1, 3, 5]                   |   ['a', 'b', 'c']
        """
        node = self.head

        str_rep = "["
        while node is not None:
            str_rep += str(node)
            if node.next is not None:
                str_rep += ', '
            node = node.next
        str_rep += "]"

        return str_rep

    # Problem 4
    def remove(self, data):
        """Remove the first node in the list containing the data.

        Raises:
            ValueError: if the list is empty or does not contain the data.

        Examples:
            print(l1)               |   >>> print(l2)
            ['a', 'e', 'i', 'o', 'u']   |   [2, 4, 6, 8]
            l1.remove('i')          |   >>> l2.remove(10)
            l1.remove('a')          |   ValueError: <message>
            l1.remove('u')          |   >>> l3 = LinkedList()
            print(l1)               |   >>> l3.remove(10)
            ['e', 'o']                  |   ValueError: <message>
        """
        # Find the node to remove
        node = self.find(data)

        # Special Cases for head and tail
        if node is self.head and node is self.tail:
            self.head = None
        elif node is self.head:
            self.head = self.head.next
        elif node is self.tail:
            self.tail = self.tail.prev

        # General Case for remove
        else:
            prev_node = node.prev
            next_node = node.next
            prev_node.next = next_node
            next_node.prev = prev_node

        self.size -= 1

    # Problem 5
    def insert(self, index, data):
        """Insert a node containing data into the list immediately before the
        node at the index-th location.

        Raises:
            IndexError: if index is negative or strictly greater than the
                current number of nodes.

        Examples:
            print(l1)               |   >>> len(l2)
            ['b']                       |   5
            l1.insert(0, 'a')       |   >>> l2.insert(6, 'z')
            print(l1)               |   IndexError: <message>
            ['a', 'b']                  |
            l1.insert(2, 'd')       |   >>> l3 = LinkedList()
            print(l1)               |   >>> l3.insert(1, 'a')
            ['a', 'b', 'd']             |   IndexError: <message>
            l1.insert(2, 'c')       |
            print(l1)               |
            ['a', 'b', 'c', 'd']        |
        """

        if index == self.size:
            # Special case of appending the node to end
            self.append(data)
        else:
            node = self.get(index)
            new_node = LinkedListNode(data)

            # Case for inserting at the start
            if node is self.head:
                self.head = new_node
                new_node.next = node
                node.prev = new_node

            # General case for inserting
            else:
                prev_node = node.prev
                prev_node.next = new_node
                new_node.prev = node.prev
                new_node.next = node
                node.prev = new_node

            self.size += 1


# Problem 6: Deque class.
class Deque(LinkedList):
    """Deque data structure class that inherits from LinkedList.

        Attributes from Linkedlist:
            head (LinkedListNode): the first node in the list.
            tail (LinkedListNode): the last node in the list.
        """

    def __init__(self):
        """Initialize the head and tail attributes by setting
        them to None via LinkedList constructor.
        """
        LinkedList.__init__(self)

    def pop(self):
        """Remove the last node and return its data"""

        # If length of list is 0 raise error
        if self.size == 0:
            raise ValueError("pop() - list is empty")

        ret_val = self.tail.value

        # Special case if size is 1
        if self.size == 1:
            self.head = None
        else:
            node = self.tail
            self.tail = self.tail.prev
            self.tail.next = None
            node.prev = None

        self.size -= 1
        return ret_val

    def popleft(self):
        """Remove the first node and return its data"""
        if self.size == 0:
            raise ValueError("popleft() - list is empty")

        ret_val = self.head.value
        self.remove(self.head.value)
        return ret_val

    def appendleft(self, data):
        """Insert a new node at the beginning of the list"""
        self.insert(0, data)

    def remove(*args, **kwargs):
        """Disabling the LinkedList remove function"""
        raise NotImplementedError("Use pop() or popleft() for removal")

    def insert(*args, **kwargs):
        """Disabling the LinkedList insert function"""
        raise NotImplementedError("Use append() or appendleft() for removal")


# Problem 7
def prob7(infile, outfile):
    """Reverse the contents of a file by line and write the results to
    another file.

    Parameters:
        infile (str): the file to read from.
        outfile (str): the file to write to.
    """
    deque = Deque()
    with open(infile, 'r') as infile:
        contents = infile.read().split('\n')
        for line in contents:
            deque.append(line)

    with open(outfile, 'w') as outfile:
        for i in range(len(deque)):
            outfile.write(deque.pop() + '\n')


@pytest.fixture
def set_up_linked_list():
    "Fixture to setup linked list"
    linked_list = LinkedList()
    for data in (2, 4, 6, 8):
        linked_list.append(data)
    return linked_list


def test_node():
    """Test method for Node"""
    with pytest.raises(TypeError) as excinfo:
        Node(Node(5))
    assert excinfo.value.args[0] == "Tell me why, data ain't type int, float, or str... I want it that way"


def test_linked_list_find(set_up_linked_list):
    """Test linked list find method"""
    linked_list = set_up_linked_list
    assert linked_list.size == 4, "Check size of linked list with length 4"
    assert linked_list.find(4).value == 4, "Find the node with value 4"
    with pytest.raises(ValueError) as excinfo:
        linked_list.find(100)
    assert excinfo.value.args[0] == "find() - Oh deary me, I'm afraid no such node exists"


def test_linked_list_get(set_up_linked_list):
    """Test linked list get method"""
    linked_list = set_up_linked_list
    assert linked_list.get(0).value == 2, "Check first value in list"
    assert linked_list.get(linked_list.size - 1).value == 8, "Check last value in list"
    with pytest.raises(IndexError) as excinfo:
        linked_list.get(-1)
    assert excinfo.value.args[0] == "get() - i appears to be negative or greater than the size of the list"
    with pytest.raises(IndexError) as excinfo:
        linked_list.get(linked_list.size)
    assert excinfo.value.args[0] == "get() - i appears to be negative or greater than the size of the list"


def test_linked_list_len(set_up_linked_list):
    """Test linked list __len__ method"""
    linked_list = set_up_linked_list
    assert linked_list.size == len(linked_list), "Checking length with size of list"


def test_linked_list_str(set_up_linked_list):
    """Test linked list __str__ method"""
    # Setup initial lists
    linked_list = set_up_linked_list
    empty_list = LinkedList()
    str_list = LinkedList()
    for char in ('2', '4', '6', '8'):
        str_list.append(char)
    quote_list = LinkedList()
    for word in ("ain't", "ain't", "a", "word"):
        quote_list.append(word)

    assert str(linked_list) == str([2, 4, 6, 8]), "Checking str function of list"
    assert str(empty_list) == str([]), "Checking str function of empty list"
    assert str(str_list) == str(['2', '4', '6', '8']), "Checking str function of str list"
    assert str(quote_list) == str(["ain't", "ain't", "a", "word"]), "Checking str function of words with quotes in them"


def test_linked_list_remove(set_up_linked_list):
    """Test linked list remove method"""
    linked_list = set_up_linked_list
    linked_list.remove(2)
    assert linked_list.head.value == 4, "Checking if head has updated"
    assert linked_list.head.next.value == 6, "Checking if head next has updated"
    assert linked_list.size == 3, "Checking size of list after remove"
    linked_list.remove(8)
    assert linked_list.tail.value == 6, "Checking if head has updated"
    assert linked_list.tail.prev.value == 4, "Checking if tail prev has updated"
    assert linked_list.size == 2, "Checking size of list after remove"
    with pytest.raises(ValueError) as excinfo:
        linked_list.remove(100)
    assert excinfo.value.args[0] == "find() - Oh deary me, I'm afraid no such node exists"


def test_linked_list_insert(set_up_linked_list):
    """Test linked list insert method"""
    linked_list = set_up_linked_list
    linked_list.insert(linked_list.size, 12)
    assert linked_list.tail.value == 12, "Checking an insert at the end of a list"
    assert len(linked_list) == 5, "Checking size of list after insert"
    linked_list.insert(0, 0)
    assert linked_list.head.value == 0, "Checking an insert at the start of a list"
    assert len(linked_list) == 6, "Checking size of list after insert"
    linked_list.insert(2, 3)
    assert linked_list.get(2).value == 3, "Checking general insert into list"
    assert len(linked_list) == 7


@pytest.fixture
def set_up_deque():
    "Fixture to setup deque"
    deque = Deque()
    for data in (2, 4, 6, 8):
        deque.append(data)
    return deque


def test_deque_pop(set_up_deque):
    """Test deque pop method"""
    deque = set_up_deque
    assert deque.pop() == 8, "Check general pop"
    assert len(deque) == 3, "Check length decrease"
    deque.pop()
    deque.pop()
    assert deque.pop() == 2, "Check pop of last element"
    with pytest.raises(ValueError) as excinfo:
        deque.pop()
    assert excinfo.value.args[0] == "pop() - list is empty"


def test_deque_popleft(set_up_deque):
    """Test deque popleft method"""
    deque = set_up_deque
    assert deque.popleft() == 2, "Check popleft of deque"
    assert len(deque) == 3, "Check length decrease"
    deque.popleft()
    deque.popleft()
    deque.popleft()
    with pytest.raises(ValueError) as excinfo:
        deque.popleft()
    assert excinfo.value.args[0] == "popleft() - list is empty"


def test_deque_appendleft(set_up_deque):
    """Test deque appendleft method"""
    deque = set_up_deque
    deque.appendleft(0)
    assert deque.size == 5, "Check length increase"
    assert deque.head.value == 0, "Check appendleft case"


def test_deque_disabled_functions(set_up_deque):
    """Test deque disabled methods"""
    deque = set_up_deque
    with pytest.raises(NotImplementedError) as excinfo:
        deque.remove(2)
    assert excinfo.value.args[0] == "Use pop() or popleft() for removal"
    with pytest.raises(NotImplementedError) as excinfo:
        deque.insert(2, 3)
    assert excinfo.value.args[0] == "Use append() or appendleft() for removal"


def test_deque_fileIO():
    prob7("in.txt", "out.txt")
