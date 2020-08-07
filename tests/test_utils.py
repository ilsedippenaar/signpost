from typing import Any, List

import pytest

from signpost.utils import Wrappable, wrap


@pytest.mark.parametrize(
    "labels, expected",
    [
        (["foo", "bar"], ["foo", "bar"]),
        ("foo", ["foo"]),
        (1, [1]),
        ((1, "foo"), [(1, "foo")]),
        ([1, "foo"], [1, "foo"]),
        ([], []),
    ],
)
def test_wrap(labels: Wrappable[Any], expected: List[Any]) -> None:
    assert wrap(labels) == expected
