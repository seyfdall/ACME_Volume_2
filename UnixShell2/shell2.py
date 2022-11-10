# shell2.py
"""Volume 3: Unix Shell 2.
<Name> Dallin Seyfried
<Class> Volume 2 Lab 002
<Date> 11/10
"""

import os
from glob import glob
import numpy as np
import subprocess

# Problem 3
def grep(target_string, file_pattern):
    """Find all files in the current directory or its subdirectories that
    match the file pattern, then determine which ones contain the target
    string.

    Parameters:
        target_string (str): A string to search for in the files whose names
            match the file_pattern.
        file_pattern (str): Specifies which files to search.

    Returns:
        matched_files (list): list of the filenames that matched the file
               pattern AND the target string.
    """
    ret_files = []
    # Get the matched files with names that match the file_pattern
    matched_files = glob("**/" + file_pattern, recursive=True)

    # Cycle through matched_files and check if it contains the string
    for file_name in matched_files:
        with open(file_name, 'r') as file:
            if target_string in file.read():
                ret_files.append(file_name)

    return ret_files

# Problem 4
def largest_files(n):
    """Return a list of the n largest files in the current directory or its
    subdirectories (from largest to smallest).
    """
    file_names = glob("**/*", recursive=True)
    file_sizes = []
    for file_name in file_names:
        file_sizes.append(os.path.getsize(file_name))

    order = np.argsort(file_sizes)[::-1]
    file_names = np.array(file_names)[order]
    smallest_file_name = file_names[n - 1]
    subprocess.Popen(["wc -l < " + smallest_file_name + " > smallest.txt"], shell=True)
    return file_names[:n]

    
# Problem 6    
def prob6(n = 10):
   """this problem counts to or from n three different ways, and
      returns the resulting lists each integer
   
   Parameters:
       n (int): the integer to count to and down from
   Returns:
       integerCounter (list): list of integers from 0 to the number n
       twoCounter (list): list of integers created by counting down from n by two
       threeCounter (list): list of integers created by counting up to n by 3
   """
   #print what the program is doing
   integerCounter = list()
   twoCounter = list()
   threeCounter = list()
   counter = n
   for i in range(n+1):
       integerCounter.append(i)
       if (i % 2 == 0):
           twoCounter.append(counter - i)
       if (i % 3 == 0):
           threeCounter.append(i)
   #return relevant values
   return integerCounter, twoCounter,


def test_grep():
    print(grep("bash", "*.txt"))


def test_largest_files():
    print(largest_files(10))


if __name__ == "__main__":
    print(largest_files(10))
