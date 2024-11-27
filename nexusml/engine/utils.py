import random
import string


def get_random_string(length: int) -> str:
    """
    Function that generates a random string of the given length for using it as a temp file names

    Args:
        length (int): the number of characters

    Returns:
        str: The random string
    """
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str
