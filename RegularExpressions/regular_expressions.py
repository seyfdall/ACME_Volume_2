# regular_expressions.py
"""Volume 3: Regular Expressions.
<Name> Dallin Seyfried
<Class> 001
<Date> 02/20/2023
"""

import re

# Problem 1
def prob1():
    """Compile and return a regular expression pattern object with the
    pattern string "python".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    # Using re.compile make a pattern object for finding python
    return re.compile("python")


# Test problem 1
def test_prob1():
    pattern = prob1()
    print(pattern.search("pypythonthon"))


# Problem 2
def prob2():
    """Compile and return a regular expression pattern object that matches
    the string "^{@}(?)[%]{.}(*)[_]{&}$".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile(r"\^\{@\}\(\?\)\[%\]\{\.\}\(\*\)\[_\]\{&\}\$")


# Test problem 2
def test_prob2():
    pattern = prob2()
    print(pattern.search("123^{@}(?)[%]{.}(*)[_]{&}$"))


# Problem 3
def prob3():
    """Compile and return a regular expression pattern object that matches
    the following strings (and no other strings).

        Book store          Mattress store          Grocery store
        Book supplier       Mattress supplier       Grocery supplier

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile(r"^(Book|Mattress|Grocery) (store|supplier)$")


# Test Problem 3
def test_prob3():
    pattern = prob3()
    print(pattern.search("Book store"))
    print(pattern.search("books Book store"))


# Problem 4
def prob4():
    """Compile and return a regular expression pattern object that matches
    any valid Python identifier.

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile(r"^[_a-zA-Z]\w*\s*((=\s*\d*\.\d*)|(=\s*'[^']*')|(=\s*[_a-zA-Z]\w*))?$")


# Test Problem 4
def test_prob4():
    pattern = prob4()
    print('\n')

    # Success tests
    for test in ["Mouse", "_num = 2.3", "arg_ = 'hey'", "__x__", "var24",
                 "max=total", "string= ''", "num_guesses"]:
        print(test + ':', bool(pattern.search(test)))

    print('\n')

    # Fail tests
    for test in ["3rats", "_num = 2.3.2", "arg_ = 'one'two", "sq(x)", " x",
                 "max=2total", "is_4=(value==4)", "pattern = r'^one|two fish$'"]:
        print(test + ':', bool(pattern.search(test)))


# Problem 5
def prob5(code):
    """Use regular expressions to place colons in the appropriate spots of the
    input string, representing Python code. You may assume that every possible
    colon is missing in the input string.

    Parameters:
        code (str): a string of Python code without any colons.

    Returns:
        (str): code, but with the colons inserted in the right places.
    """
    # Replacement function to add colon at end of block
    def append_colon(match_obj):
        if match_obj.group() is not None:
            return match_obj.group() + ':'

    # Search through string and append colons to get appropriate python block
    return re.sub(
        pattern=r"^\s*(((if|elif|for|while|try|except|with|def|class).*)|(else|finally|except))+?$",
        repl=append_colon,
        string=code,
        flags=re.MULTILINE
    )


# Test Problem 5
def test_prob5():
    block = """"
            k, i, p = 999, 1, 0
            while k > i
                i *= 2
                p += 1
                if k != 999
                    61
                    print("k should not have changed")
                else
                    pass
            print(p)
            """
    print(prob5(block))


# Problem 6
def prob6(filename="fake_contacts.txt"):
    """Use regular expressions to parse the data in the given file and format
    it uniformly, writing birthdays as mm/dd/yyyy and phone numbers as
    (xxx)xxx-xxxx. Construct a dictionary where the key is the name of an
    individual and the value is another dictionary containing their
    information. Each of these inner dictionaries should have the keys
    "birthday", "email", and "phone". In the case of missing data, map the key
    to None.

    Returns:
        (dict): a dictionary mapping names to a dictionary of personal info.
    """

    # Read in text from filename for every line
    file = open(filename, mode='r')
    text = file.readlines()
    file.close()

    contacts = dict()

    # pattern to find name
    name_pattern = re.compile(r"^[a-zA-Z]* ([A-Z]\. )?[a-zA-Z]*")

    # pattern to find birthday
    birthday_pattern = re.compile(r"\d{1,2}/\d{1,2}/((\d\d\d\d)|(\d\d))")

    # Function to clean birthday
    def clean_birthday_info(birth):
        if birth is not None:
            # birth_pat = re.compile("(^[0-9]{1,2}/) (/[0-9]{1,2}/) (/[0-9]{2}$)|(/[0-9]{4}$)")
            parts = re.split(r"/", birth)
            if len(parts[0]) == 1:
                parts[0] = '0' + parts[0]
            if len(parts[1]) == 1:
                parts[1] = '0' + parts[1]
            if len(parts[2]) == 2:
                parts[2] = '20' + parts[2]
            birth = parts[0] + '/' + parts[1] + '/' + parts[2]
        return birth

    # pattern to find email
    email_pattern = re.compile(r"[a-zA-Z0-9_\.]*@[a-zA-Z0-9_\.]*")

    # pattern to find phone number
    phone_pattern = re.compile(r"(\d-)?((\(\d{3}\)-?)|(\d{3}-))\d{3}-\d{4}")

    # Function to clean phone number
    def clean_number_info(number):
        if number is not None:
            number = re.sub(r"^\d-", "", number)
            if number[0] != "(":
                number = "(" + number
            if number[4] != ")":
                number = number[:4] + ")" + number[4:]
            number = re.sub(r"(\)-?)", ")", number)
        return number

    for info in text:
        # Get the name which could include middle initial
        name = name_pattern.search(info).group()

        # Get the birthday if it exists
        match_birth = birthday_pattern.search(info)
        birthday = None if match_birth is None else match_birth.group()
        birthday = clean_birthday_info(birthday)

        # Get the email if it exists
        match_email = email_pattern.search(info)
        email = None if match_email is None else match_email.group()

        # Get the phone number if it exists
        match_phone = phone_pattern.search(info)
        phone = None if match_phone is None else match_phone.group()
        phone = clean_number_info(phone)

        contacts[name] = dict(birthday=birthday, email=email, phone=phone)

    return contacts


# Test Problem 6
def test_prob6():
    dictionary = prob6()
    print(dictionary)
    print('\n')
    print(dictionary.get("John Doe"))


