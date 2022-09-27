# test_specs.py
"""Python Essentials: Unit Testing.
<Name> Dallin Seyfried
<Class> Math 321
<Date> 9/22/2022
"""

import specs
import pytest


def test_add():
    """Function to test the addition function for fraction"""
    assert specs.add(1, 3) == 4, "failed on positive integers"
    assert specs.add(-5, -7) == -12, "failed on negative integers"
    assert specs.add(-6, 14) == 8


def test_divide():
    """Function to test division function for fraction"""
    assert specs.divide(4,2) == 2, "integer division"
    assert specs.divide(5,4) == 1.25, "float division"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.divide(4, 0)
    assert excinfo.value.args[0] == "second input cannot be zero"


# Problem 1: write a unit test for specs.smallest_factor(), then correct it.
def test_smallest_factor():
    """Tests the smallest factor function - problem found with 6 and 2"""
    assert specs.smallest_factor(1) == 1, "1 and 1"
    assert specs.smallest_factor(6) == 2, "6 and 2"
    assert specs.smallest_factor(9) == 3, "9 and 3"
    assert specs.smallest_factor(21) == 3, "21 and 3"
    assert specs.smallest_factor(13) == 13, "13 and 13"


# Problem 2: write a unit test for specs.month_length().
def test_month_length():
    """Testing the month_length function"""
    assert specs.month_length("Dodecatember", True) is None, "Not a month test"
    assert specs.month_length("September", False) == 30, "Month with 30 days"
    assert specs.month_length("January", False) == 31, "Month with 31 days"
    assert specs.month_length("February", False) == 28, "February not on leap year"
    assert specs.month_length("February", True) == 29, "February on a leap year"


# Problem 3: write a unit test for specs.operate().
def test_operate():
    """Testing the operate function"""
    with pytest.raises(TypeError) as excinfo:
        specs.operate(1, 2, 5 % 5)
    assert specs.operate(5, 5, '+') == 10, "Addition Case"
    assert specs.operate(5, 5, '-') == 0, "Subtraction Case"
    assert specs.operate(5, 5, '*') == 25, "Multiplication Case"
    assert specs.operate(5, 5, '/') == 1, "Division Case"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.operate(1, 0, '/')
    with pytest.raises(ValueError) as excinfo:
        specs.operate(1, 1, '5')

# Problem 4: write unit tests for specs.Fraction, then correct it.
@pytest.fixture
def set_up_fractions():
    """Fixture to set up initial fractions to be used in later tests"""
    frac_1_3 = specs.Fraction(1, 3)
    frac_1_2 = specs.Fraction(1, 2)
    frac_n2_3 = specs.Fraction(-2, 3)
    return frac_1_3, frac_1_2, frac_n2_3


def test_fraction_init(set_up_fractions):
    """Unit test to test fraction constructor"""
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_3.numer == 1
    assert frac_1_2.denom == 2
    assert frac_n2_3.numer == -2
    frac = specs.Fraction(30, 42)
    assert frac.numer == 5
    assert frac.denom == 7
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.Fraction(1, 0)
    with pytest.raises(TypeError) as excinfo:
        specs.Fraction("2", "5")


def test_fraction_str(set_up_fractions):
    """Unit test to test the fraction str method"""
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert str(frac_1_3) == "1/3"
    assert str(frac_1_2) == "1/2"
    assert str(frac_n2_3) == "-2/3"
    assert str(specs.Fraction(1, 1)) == "1"


def test_fraction_float(set_up_fractions):
    """Unit test to test the fraction float operation"""
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert float(frac_1_3) == 1 / 3.
    assert float(frac_1_2) == .5
    assert float(frac_n2_3) == -2 / 3.


def test_fraction_eq(set_up_fractions):
    """Unit test to test Fraction equality method"""
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 == specs.Fraction(1, 2)
    assert frac_1_3 == specs.Fraction(2, 6)
    assert frac_n2_3 == specs.Fraction(8, -12)
    assert frac_1_2 == 0.5


def test_fraction_add(set_up_fractions):
    """Function to test fraction's add function problem
    found with adding fractions of different bases together"""
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert str(frac_1_3 + frac_1_2) == "5/6"
    assert str(frac_1_3 + frac_n2_3) == "-1/3"


def test_fraction_sub(set_up_fractions):
    """Function to test fraction's sub function problem
    found with subtracting fractions of different bases together"""
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert str(frac_1_3 - frac_1_2) == "-1/6"
    assert str(frac_1_3 - frac_n2_3) == "1"


def test_fraction_mul(set_up_fractions):
    """Function to test fraction's mul function"""
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert str(frac_1_3 * frac_1_2) == "1/6"
    assert str(frac_1_3 * frac_n2_3) == "-2/9"


def test_fraction_truediv(set_up_fractions):
    """Function to test fraction's truediv function"""
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert str(frac_1_3 / frac_1_2) == "2/3"
    assert str(frac_1_3 / frac_n2_3) == "-1/2"
    with pytest.raises(ZeroDivisionError) as excinfo:
        frac_1_2 / specs.Fraction(0, 1)

# Problem 5: Write test cases for Set.
def test_count_sets():
    """Function to test the count sets function"""
    # Test a good hand
    assert specs.count_sets(["1022", "1111", "1200", "0010",
                             "2201", "2111", "0020", "0000",
                            "1102", "0210", "2110", "1020"]) == 4
    # Test invalid amount of cards
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(["1000", "2000", "0111"])
    assert excinfo.value.args[0] == "There are not exactly 12 cards"

    # Test non digit input
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(["102a", "1111", "1200", "0010",
                          "2201", "2111", "0020", "0000",
                          "1102", "0210", "2110", "1020"])
    assert excinfo.value.args[0] == "One or more cards has a character other than 0, 1, or 2"

    # Test invalid length of cards
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(["10234", "1111", "1200", "0010",
                          "2201", "2111", "0020", "0000",
                          "1102", "0210", "2110", "1020"])
    assert excinfo.value.args[0] == "One or more cards does not have exactly 4 digits"

    # Test uniqueness
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(["1023", "1023", "1200", "0010",
                          "2201", "2111", "0020", "0000",
                          "1102", "0210", "2110", "1020"])
    assert excinfo.value.args[0] == "The cards are not all unique"

    # Test another good hand
    assert specs.count_sets(["1022", "1122", "0100", "2021",
                            "0010", "2201", "2111", "0020",
                            "1102", "0210", "2110", "1020"]) == 6


def test_is_set():
    """Function to test if a group of cards is a set"""
    assert specs.is_set("1022", "1111", "1200")
    assert not specs.is_set("1000", "2000", "0111")
    assert specs.is_set("0000", "1000", "2000")