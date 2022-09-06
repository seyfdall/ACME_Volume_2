# python_intro.py
"""Python Essentials: Introduction to Python.
<Name> Dallin Seyfried
<Class> Section 2
<Date> 08/13/22
"""


# Problem 2
def sphere_volume(r):
    """ Return the volume of the sphere of radius 'r'.
    Use 3.14159 for pi in your computation.
    """
    # Calculate and return the volume of a sphere => 4/3 * pi * r^2
    return (4 / 3) * 3.14159 * r**3


# Problem 3
def isolate(a, b, c, d, e):
    """ Print the arguments separated by spaces, but print 5 spaces on either
    side of b.
    """
    # Print arguments separated by 5 spaces between a, b, and c then a normal space after c
    print(a, b, c, sep='     ', end=' ')
    # Print d and e with a space separating them.
    print(d, e, sep=' ')


# Problem 4
def first_half(my_string):
    """ Return the first half of the string 'my_string'. Exclude the
    middle character if there are an odd number of characters.

    Examples:
        >>> first_half("python")
        'pyt'
        >>> first_half("ipython")
        'ipy'
    """
    # Calculates the length of the first half of the string using integer
    #   division to ignore the middle character if there are an odd # of characters
    half_len = len(my_string) // 2
    return my_string[:half_len]

def backward(my_string):
    """ Return the reverse of the string 'my_string'.

    Examples:
        >>> backward("python")
        'nohtyp'
        >>> backward("ipython")
        'nohtypi'
    """
    # Using python list functionality reverses the string
    return my_string[::-1]


# Problem 5
def list_ops():
    """ Define a list with the entries "bear", "ant", "cat", and "dog".
    Perform the following operations on the list:
        - Append "eagle".
        - Replace the entry at index 2 with "fox".
        - Remove (or pop) the entry at index 1.
        - Sort the list in reverse alphabetical order.
        - Replace "eagle" with "hawk".
        - Add the string "hunter" to the last entry in the list.
    Return the resulting list.

    Examples:
        >>> list_ops()
        ['fox', 'hawk', 'dog', 'bearhunter']
    """
    # Initialize the list
    animals = ['bear', 'ant', 'cat', 'dog']
    # Append the word 'eagle' to the list
    animals.append('eagle')
    # Set the third word to be 'fox'
    animals[2] = 'fox'
    # Remove the second string of the list
    animals.pop(1)
    # Reverse the list
    animals.sort(reverse=True)
    # Replace 'eagle' with 'hawk'
    animals[animals.index('eagle')] = 'hawk'
    # Concatenating the word 'hunter' to the last string in the list
    animals[len(animals) - 1] = animals[len(animals) - 1] + 'hunter'
    return animals


# Problem 6
def pig_latin(word):
    """ Translate the string 'word' into Pig Latin, and return the new word.

    Examples:
        >>> pig_latin("apple")
        'applehay'
        >>> pig_latin("banana")
        'ananabay'
    """
    # Create a string of vowels to check with the starting letter
    vowels = 'aeiou'
    # If the starting letter is a vowel add 'hay' to the word
    if word[0].lower() in vowels:
        word = word + 'hay'
    # If the starting letter is not a vowel, move the first letter
    #   to after the word and then add 'ay'
    else:
        word = word + word[0]
        word = word[1:]
        word = word + 'ay'
    return word


# Problem 7
def palindrome():
    """ Find and retun the largest panindromic number made from the product
    of two 3-digit numbers.
    """
    panindromic_number = 0
    # Cycle through each 3-digit number twice to check for panindromic numbers
    for i in range(100, 999):
        for j in range(100, 999):
            # Generate the test number
            new_num = i * j
            # Stringify the test number to check later
            num_str = str(new_num)
            isPanindromic = True
            # Cycle through each character to compare with the reverse
            #   to see if it's panindromic
            for k, char in enumerate(reversed(num_str)):
                if char != num_str[k]:
                    isPanindromic = False
                    break
            # If the number is a panindrome and if it's less than the
            #   current largest then set it as the new panindromic_number
            if isPanindromic and panindromic_number < new_num:
                panindromic_number = new_num
    return panindromic_number


# Problem 8
def alt_harmonic(n):
    """ Return the partial sum of the first n terms of the alternating
    harmonic series, which approximates ln(2).
    """
    # Using list comprehension to generate the approximation
    return sum([((-1)**(i+1))/i for i in range(1, n + 1)])


# Problem 1
if __name__ == "__main__":
    print("Hello World")
