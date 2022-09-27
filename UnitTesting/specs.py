# specs.py
"""Python Essentials: Unit Testing.
<Name> Dallin Seyfried
<Class> Math 321
<Date> 9/22/2022
"""

from itertools import combinations


def add(a, b):
    """Add two numbers."""
    return a + b

def divide(a, b):
    """Divide two numbers, raising an error if the second number is zero."""
    if b == 0:
        raise ZeroDivisionError("second input cannot be zero")
    return a / b


# Problem 1
def smallest_factor(n):
    """Return the smallest prime factor of the positive integer n."""
    if n == 1: return 1
    for i in range(2, int(n**.5 + 1)): # Corrected problem by adding 1 to the truncated total
        if n % i == 0: return i
    return n


# Problem 2
def month_length(month, leap_year=False):
    """Return the number of days in the given month."""
    if month in {"September", "April", "June", "November"}:
        return 30
    elif month in {"January", "March", "May", "July",
                        "August", "October", "December"}:
        return 31
    # Edge cases for February
    if month == "February":
        if not leap_year:
            return 28
        else:
            return 29
    else:
        return None


# Problem 3
def operate(a, b, oper):
    """Apply an arithmetic operation to a and b."""
    if type(oper) is not str:
        raise TypeError("oper must be a string")
    elif oper == '+':
        return a + b
    elif oper == '-':
        return a - b
    elif oper == '*':
        return a * b
    elif oper == '/':
        if b == 0:
            raise ZeroDivisionError("division by zero is undefined")
        return a / b
    raise ValueError("oper must be one of '+', '/', '-', or '*'")


# Problem 4
class Fraction(object):
    """Reduced fraction class with integer numerator and denominator."""
    def __init__(self, numerator, denominator):
        """Constructor to initialize numerator and denominator"""
        if denominator == 0:
            raise ZeroDivisionError("denominator cannot be zero")
        elif type(numerator) is not int or type(denominator) is not int:
            raise TypeError("numerator and denominator must be integers")

        # Function to calculate gcd
        def gcd(a,b):
            while b != 0:
                a, b = b, a % b
            return a
        common_factor = gcd(numerator, denominator)
        self.numer = numerator // common_factor
        self.denom = denominator // common_factor

    def __str__(self):
        """Magic function to return a formated string describing the fraction"""
        if self.denom != 1:
            return "{}/{}".format(self.numer, self.denom)
        else:
            return str(self.numer)

    def __float__(self):
        """Magic function to return the floating point value of the fraction"""
        return self.numer / self.denom

    def __eq__(self, other):
        """Magic function to do equality"""
        if type(other) is Fraction:
            return self.numer==other.numer and self.denom==other.denom
        else:
            return float(self) == other

    def __add__(self, other):
        """Magic function to do addition"""
        # Problem found here
        return Fraction(self.denom*other.numer + self.numer*other.denom,
                                                        self.denom*other.denom)

    def __sub__(self, other):
        """Magic function to do subtraction"""
        # Problem found here
        return Fraction(self.numer*other.denom - self.denom*other.numer,
                                                        self.denom*other.denom)

    def __mul__(self, other):
        """Magic function to do multiplication"""
        return Fraction(self.numer*other.numer, self.denom*other.denom)

    def __truediv__(self, other):
        """Magic function to do division"""
        if self.denom*other.numer == 0:
            raise ZeroDivisionError("cannot divide by zero")
        return Fraction(self.numer*other.denom, self.denom*other.numer)


# Problem 6
def count_sets(cards):
    """Return the number of sets in the provided Set hand.

    Parameters:
        cards (list(str)) a list of twelve cards as 4-bit integers in
        base 3 as strings, such as ["1022", "1122", ..., "1020"].
    Returns:
        (int) The number of sets in the hand.
    Raises:
        ValueError: if the list does not contain a valid Set hand, meaning
            - there are not exactly 12 cards,
            - the cards are not all unique,
            - one or more cards does not have exactly 4 digits, or
            - one or more cards has a character other than 0, 1, or 2.
    """

    # Checking to see if there are 12 cards
    if len(cards) != 12:
        raise ValueError("There are not exactly 12 cards")

    # Checking uniqueness in the hand
    for card1 in cards:
        count = 0
        for card2 in cards:
            if card1 == card2:
                count += 1
            if count > 1:
                raise ValueError("The cards are not all unique")

    # Check to make sure each card has 4 digits and each digit is either 0, 1, or 2
    for card in cards:
        if len(card) != 4:
            raise ValueError("One or more cards does not have exactly 4 digits")
        for i in range(len(card)):
            if card[i] not in ['0', '1', '2']:
                raise ValueError("One or more cards has a character other than 0, 1, or 2")

    # Calculate total sets using the helper function is_set
    total_sets = 0
    for three_cards in list(combinations(cards, 3)):
        if is_set(three_cards[0], three_cards[1], three_cards[2]):
            total_sets += 1
    return total_sets


def is_set(a, b, c):
    """Determine if the cards a, b, and c constitute a set.

    Parameters:
        a, b, c (str): string representations of 4-bit integers in base 3.
            For example, "1022", "1122", and "1020" (which is not a set).
    Returns:
        True if a, b, and c form a set, meaning the ith digit of a, b,
            and c are either the same or all different for i=1,2,3,4.
        False if a, b, and c do not form a set.
    """
    # Cycle through each digit in each card and make sure they meet requirements
    for i in range(4):
        if (int(a[i]) + int(b[i]) + int(c[i])) % 3 != 0:
            return False
    return True
