import itertools
from typing import Any, Collection, Dict, Hashable, Optional, Set, cast

import numpy as np
import pandas as pd
import pytest
from _pytest.fixtures import SubRequest

from signpost import (
    Bounded,
    Cols,
    MergeResult,
    Notna,
    Schema,
    Superkey,
    Values,
)
from signpost import properties as props


@pytest.fixture
def basic_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["foo", "bar", "bar"]})


@pytest.fixture
def types_df() -> pd.DataFrame:
    """
    Straight from the docs
    """
    return pd.DataFrame(
        {
            "a": pd.Series([1, 2, 3], dtype=np.dtype("int32")),
            "b": pd.Series(["x", "y", "z"], dtype=np.dtype("O")),
            "c": pd.Series([True, False, np.nan], dtype=np.dtype("O")),
            "d": pd.Series(["h", "i", np.nan], dtype=np.dtype("O")),
            "e": pd.Series([10, np.nan, 20], dtype=np.dtype("float")),
            "f": pd.Series([np.nan, 100.5, 200], dtype=np.dtype("float")),
            "g": pd.Series([1, 2, 3], dtype=np.dtype("int64")),
        }
    )


@pytest.fixture
def better_types_df(types_df: pd.DataFrame) -> pd.DataFrame:
    return types_df.convert_dtypes()


@pytest.fixture
def keys_df() -> pd.DataFrame:
    return pd.DataFrame(
        [[1, 1, "a", 3, 3], [2, 1, "a", np.nan, np.nan], [3, 2, "b", 4, np.nan]],
        columns=["a", "b", "c", "d", "e"],
    )


@pytest.fixture
def left_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


@pytest.fixture
def right_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [2, 3, 4], "c": [7, 8, 9]})


@pytest.fixture
def df(request: SubRequest) -> pd.DataFrame:
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["inner", "left", "right", "outer"])
def merge_how(request: SubRequest) -> str:
    return cast(str, request.param)


@pytest.fixture(
    params=[
        # power set, excluding empty set
        c
        for i in range(1, 4)
        for c in itertools.combinations(["both", "left_only", "right_only"], i)
    ]
)
def merge_results(request: SubRequest) -> Set[str]:
    return set(request.param)


@pytest.fixture(params=["_merge", "a_very_very_very_long_column_name", "_"])
def indicator_col(request: SubRequest) -> str:
    return cast(str, request.param)


@pytest.mark.parametrize(
    "qualifier, cols, expected",
    [
        ("all", "a", True),
        ("all", ["a", "b"], True),
        ("all", ["a", "b", "c"], False),
        ("all", [], True),
        ("any", "a", True),
        ("any", ["a", "b"], True),
        ("any", ["a", "b", "c"], True),
        ("any", [], False),
        ("none", "c", True),
        ("none", ["c", "d"], True),
        ("none", ["a", "c"], False),
        ("none", [], True),
        ("just", ["a", "b"], True),
        ("just", "a", False),
        ("just", "b", False),
        ("just", [], False),
    ],
)
def test_cols(
    basic_df: pd.DataFrame, qualifier: str, cols: props.ColsType, expected: bool
) -> None:
    assert (Cols.Checker(qualifier, cols).check(basic_df) is None) == expected


@pytest.mark.parametrize(
    "df, qualifier, schema, expected",
    [
        (
            "types_df",
            "all",
            {
                "a": np.int32,
                "b": object,
                "c": object,
                "d": object,
                "e": float,
                "f": float,
                "g": int,
            },
            True,
        ),
        ("types_df", "all", {"a": np.int32}, True),
        ("types_df", "all", {"a": float}, False),
        ("types_df", "all", {"b": str}, False),
        ("types_df", "all", {"z": object}, False),
        ("types_df", "all", {}, True),
        (
            "better_types_df",
            "all",
            {
                "a": "Int32",
                "b": "string",
                "c": "boolean",
                "d": "string",
                "e": "Int64",
                "f": float,
                "g": "Int64",
            },
            True,
        ),
        ("better_types_df", "all", {"b": pd.StringDtype()}, True),
        ("better_types_df", "all", {"b": str}, False),
        ("types_df", "any", {"a": np.int32}, True),
        ("types_df", "any", {"a": float}, False),
        ("types_df", "any", {"a": np.int32, "b": int}, True),
        ("types_df", "any", {"z": object}, False),
        ("types_df", "any", {}, False),
        ("types_df", "none", {"a": np.int32}, False),
        ("types_df", "none", {"a": float}, True),
        ("types_df", "none", {"a": np.int32, "b": int}, False),
        ("types_df", "none", {"z": object}, True),
        ("types_df", "none", {}, True),
        (
            "types_df",
            "just",
            {
                "a": np.int32,
                "b": object,
                "c": object,
                "d": object,
                "e": float,
                "f": float,
                "g": int,
            },
            True,
        ),
        ("types_df", "just", {"a": np.int32}, False),
    ],
    indirect=["df"],
)
def test_schema(
    df: pd.DataFrame, qualifier: str, schema: props.SchemaType, expected: bool
) -> None:
    assert (Schema.Checker(qualifier, schema).check(df) is None) == expected


@pytest.mark.parametrize(
    "df, qualifier, values, expected",
    [
        ("basic_df", "all", {"a": [1, 2, 3]}, True),
        ("basic_df", "all", {"a": [1]}, True),
        ("basic_df", "all", {"a": [3, 4]}, False),
        ("basic_df", "all", {"a": [4]}, False),
        ("basic_df", "all", {"b": ["foo", "bar"]}, True),
        ("basic_df", "all", {"a": []}, True),
        ("basic_df", "any", {"a": [1, 2, 3]}, True),
        ("basic_df", "any", {"a": [1]}, True),
        ("basic_df", "any", {"a": [3, 4]}, True),
        ("basic_df", "any", {"a": [4]}, False),
        ("basic_df", "any", {"b": ["foo", "bar"]}, True),
        ("basic_df", "any", {"a": []}, False),
        ("basic_df", "none", {"a": [1, 2, 3]}, False),
        ("basic_df", "none", {"a": [1]}, False),
        ("basic_df", "none", {"a": [3, 4]}, False),
        ("basic_df", "none", {"a": [4]}, True),
        ("basic_df", "none", {"b": ["foo", "bar"]}, False),
        ("basic_df", "none", {"a": []}, True),
        ("basic_df", "just", {"a": [1, 2, 3]}, True),
        ("basic_df", "just", {"a": [1]}, False),
        ("basic_df", "just", {"a": [3, 4]}, False),
        ("basic_df", "just", {"a": [4]}, False),
        ("basic_df", "just", {"b": ["foo", "bar"]}, True),
        ("basic_df", "just", {"a": []}, False),
        # missingness
        ("types_df", "all", {"c": [np.nan]}, True),
        ("better_types_df", "all", {"c": [pd.NA]}, True),
        ("types_df", "any", {"c": [np.nan]}, True),
        ("better_types_df", "any", {"c": [pd.NA]}, True),
        ("types_df", "none", {"c": [np.nan]}, False),
        ("better_types_df", "none", {"c": [pd.NA]}, False),
        ("types_df", "just", {"c": [np.nan, True, False]}, True),
        ("better_types_df", "just", {"c": [pd.NA, True, False]}, True),
        # missing columns
        ("basic_df", "all", {"z": [1, 2]}, False),
        # value interactions
        ("basic_df", "all", {"a": [1, 2], "b": ["foo", "bar"]}, True),
        ("basic_df", "any", {"a": [1, -1], "b": ["foo", "bar"]}, True),
        ("basic_df", "none", {"a": [1, -1], "b": ["foo", "bar"]}, False),
        ("basic_df", "none", {"a": [1, -1], "b": ["not_foo", "bar"]}, True),
    ],
    indirect=["df"],
)
def test_values(
    df: pd.DataFrame,
    qualifier: str,
    values: Dict[Hashable, Collection[Any]],
    expected: bool,
) -> None:
    assert (Values.Checker(qualifier, values).check(df) is None) == expected


@pytest.mark.parametrize(
    "cols, over, expected",
    [
        ("a", None, True),
        ("b", None, False),
        ("d", None, True),  # NaNs are counted as values
        ("e", None, False),  # duplicate NaNs are counted as duplicates
        ("a", "b", True),
        ("a", ["b", "c"], True),
        ("b", "c", True),
        ("b", "a", False),
        ("b", "e", False),
        ("b", ["c", "e"], False),
        ("c", "b", True),  # reverse of b and c above
        ("d", "a", True),  # NaNs can be keys
        ([], None, False),  # empty list is not a superkey
        # missing columns
        ("z", [], False),
        ("a", "z", False),
        ("x", ["y", "z"], False),
    ],
)
def test_superkey(
    keys_df: pd.DataFrame,
    cols: props.ColsType,
    over: Optional[props.ColsType],
    expected: bool,
) -> None:
    assert (Superkey.Checker(cols, over=over).check(keys_df) is None) == expected


@pytest.mark.parametrize(
    "df, qualifier, cols, expected",
    [
        ("types_df", "all", "a", True),
        ("types_df", "all", "c", False),
        ("types_df", "all", ["a", "b", "g"], True),
        ("types_df", "all", ["a", "c"], False),
        ("types_df", "all", ["a", "z"], False),
        ("types_df", "any", "a", True),
        ("types_df", "any", "c", False),
        ("types_df", "any", ["a", "b", "g"], True),
        ("types_df", "any", ["a", "c"], True),
        ("types_df", "any", ["a", "z"], False),
        ("types_df", "just", "a", False),
        ("types_df", "just", "c", False),
        ("types_df", "just", ["a", "b", "g"], True),
        ("types_df", "just", ["a", "c"], False),
        ("types_df", "just", ["a", "z"], False),
        ("types_df", "none", "a", False),
        ("types_df", "none", "c", True),
        ("types_df", "none", ["a", "b", "g"], False),
        ("types_df", "none", ["a", "c"], False),
        ("types_df", "none", ["a", "z"], False),
    ],
    indirect=["df"],
)
def test_notna(
    df: pd.DataFrame, qualifier: str, cols: props.ColsType, expected: bool
) -> None:
    assert (Notna.Checker(qualifier, cols).check(df) is None) == expected


def test_merge_result(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    merge_how: str,
    merge_results: Set[str],
    indicator_col: str,
) -> None:
    expected = {
        "inner": {"both"},
        "left": {"both", "left_only"},
        "right": {"both", "right_only"},
        "outer": {"both", "left_only", "right_only"},
    }[merge_how] == merge_results
    assert (
        MergeResult.Checker(merge_results, indicator_col=indicator_col).check(
            pd.merge(left_df, right_df, how=merge_how, indicator=indicator_col)
        )
        is None
    ) == expected


@pytest.mark.parametrize(
    "cols, lower, upper, closed, expected",
    [
        ("a", 0, 4, "both", True),
        ("a", 1, 4, "both", True),
        ("a", 1, 3, "both", True),
        ("a", 1, 3, "left", False),
        ("a", 1, 3, "right", False),
        ("a", 1, 3, "neither", False),
        ("b", "x", "z", "both", True),
        (["a", "g"], 0, 4, "neither", True),
        # NA is treated as False
        ("e", -100, 100, "both", False),
        (["a", "e"], -100, 100, "both", False),
        # unbounded
        ("a", None, 4, "right", True),
        ("a", 0, None, "left", True),
        ("a", None, None, "neither", True),
        # NA not checked in double unbounded case
        (["a", "e"], None, None, "neither", True),
        # missing columns
        (["a", "foobar"], 0, 4, "both", False),
        # malformed interval
        pytest.param(
            "a",
            4,
            3,
            "both",
            True,
            marks=pytest.mark.xfail(raises=ValueError, strict=True),
        ),
        # bounded constraints
        pytest.param(
            "a",
            None,
            3,
            "both",
            True,
            marks=pytest.mark.xfail(raises=ValueError, strict=True),
        ),
        pytest.param(
            "a",
            1,
            None,
            "right",
            True,
            marks=pytest.mark.xfail(raises=ValueError, strict=True),
        ),
    ],
)
def test_bounded(
    types_df: pd.DataFrame,
    cols: props.ColsType,
    lower: Any,
    upper: Any,
    closed: str,
    expected: bool,
) -> None:
    assert (
        Bounded.Checker(cols, lower, upper, closed,).check(types_df) is None
    ) == expected
