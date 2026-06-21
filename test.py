"""Simple test module."""


def reverse_string(text):
    """Return the reversed version of a string."""
    return text[::-1]


TEST_STRING = "Hello, World!"
RESULT = reverse_string(TEST_STRING)

print(f"Original: {TEST_STRING}")
print(f"Reversed: {RESULT}")
