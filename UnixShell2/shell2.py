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
    # Get the files
    file_names = glob("**/*.*", recursive=True)
    file_sizes = []
    for file_name in file_names:
        file_sizes.append(os.path.getsize(file_name))

    # Sort by file size and write to the smallest.txt file
    order = np.argsort(file_sizes)[::-1]
    file_names = np.array(file_names)[order]
    smallest_file_name = file_names[n - 1]
    line_count = subprocess.check_output(["wc", "-l", smallest_file_name]).decode().split()[0]

    with open("smallest.txt", "w") as file:
        file.write(line_count)

    return file_names[:n]


def test_grep():
    print(grep("bash", "*.txt"))


def test_largest_files():
    print(largest_files(10))


if __name__ == "__main__":
    print(largest_files(10))
