import pytest


def add(a: int, b: int) -> int:
    return a + b


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (1, 2, 3),
        (0, 0, 0),
        (-1, 1, 0),
    ],
)
def test_add(a: int, b: int, expected: int) -> None:
    assert add(a, b) == expected