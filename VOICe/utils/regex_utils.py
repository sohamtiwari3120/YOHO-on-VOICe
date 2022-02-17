import re


def tryint(s):
    """Trying to convert input parameter into int type

    Args:
        s (Any): Input argument to be converted into string

    Returns:
        int | Any: If successful, will return int value of input argument, else returns the original argument.
    """    
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect. The sorting is done INPLACE.
    """
    l.sort(key=alphanum_key)
