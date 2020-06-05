from typing import Any, Callable, Collection, Dict, Optional

import pandas as pd
import pytest

from signpost import And, Cols, Function, Meta, Or, Schema, Superkey, Values
from signpost import properties as props


@pytest.fixture
def basic_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["foo", "bar", "bar"]})


@pytest.mark.parametrize(
    "func,context,expected",
    [((lambda d, c: None), {}, True), ((lambda d, c: ""), {}, False)],
)
def test_function(
    basic_df: pd.DataFrame,
    func: Callable[[pd.DataFrame, Dict[str, Any]], Optional[str]],
    context: Dict[str, Any],
    expected: bool,
) -> None:
    assert (Function(func).check_with_context(basic_df, context) is None) == expected


@pytest.mark.parametrize(
    "properties,expected",
    [
        ([Cols("all", ["c"]), Schema("all", {"a": float})], False),
        ([Cols("all", ["c"]), Schema("all", {"a": int})], False),
        ([Cols("all", ["a"]), Schema("all", {"a": float})], False),
        ([Cols("all", ["a"]), Schema("all", {"a": int})], True),
        ([Cols("any", ["a"])], True),
        ([], True),
    ],
)
def test_and(
    basic_df: pd.DataFrame, properties: Collection[props.Property], expected: bool,
) -> None:
    assert (And(*properties).check_with_context(basic_df, {}) is None) == expected


@pytest.mark.parametrize(
    "properties,expected",
    [
        ([Cols("all", ["c"]), Schema("all", {"a": float})], False),
        ([Cols("all", ["c"]), Schema("all", {"a": int})], True),
        ([Cols("all", ["a"]), Schema("all", {"a": float})], True),
        ([Cols("all", ["a"]), Schema("all", {"a": int})], True),
        ([Cols("any", ["a"])], True),
        ([], False),
    ],
)
def test_or(
    basic_df: pd.DataFrame, properties: Collection[props.Property], expected: bool,
) -> None:
    assert (Or(*properties).check_with_context(basic_df, {}) is None) == expected


@pytest.mark.parametrize(
    "qualifier,cols_expr,context,expected",
    [
        ("all", "cols", {"cols": ["a"]}, True),
        ("all", "cols[:1]", {"cols": ["a"], "other": ["a", "b"]}, True),
        ("all", "['a']", {}, True),
    ],
)
def test_cols(
    basic_df: pd.DataFrame,
    qualifier: str,
    cols_expr: str,
    context: Dict[str, Any],
    expected: bool,
) -> None:
    assert (
        Cols(qualifier, Meta(cols_expr)).check_with_context(basic_df, context) is None
    ) == expected


@pytest.mark.parametrize(
    "qualifier,values_expr,context,expected",
    [("all", "{c: [1] for c in cols}", {"cols": ["a"]}, True),],
)
def test_values(
    basic_df: pd.DataFrame,
    qualifier: str,
    values_expr: str,
    context: Dict[str, Any],
    expected: bool,
) -> None:
    assert (
        Values(qualifier, Meta(values_expr)).check_with_context(basic_df, context)
        is None
    ) == expected


@pytest.mark.parametrize(
    "qualifier,schema_expr,context,expected",
    [("all", "{c: int for c in cols}", {"cols": ["a"]}, True),],
)
def test_schema(
    basic_df: pd.DataFrame,
    qualifier: str,
    schema_expr: str,
    context: Dict[str, Any],
    expected: bool,
) -> None:
    assert (
        Schema(qualifier, Meta(schema_expr)).check_with_context(basic_df, context)
        is None
    ) == expected


@pytest.mark.parametrize(
    "cols_expr,over_expr,context,expected",
    [
        ("cols", "over", {"cols": ["a"], "over": ["b"]}, True),
        ("cols", "None", {"cols": ["a"], "over": ["b"]}, True),
    ],
)
def test_superkey(
    basic_df: pd.DataFrame,
    cols_expr: str,
    over_expr: str,
    context: Dict[str, Any],
    expected: bool,
) -> None:
    assert (
        Superkey(Meta(cols_expr), Meta(over_expr)).check_with_context(basic_df, context)
        is None
    ) == expected


def test_superkey_default_over(basic_df: pd.DataFrame) -> None:
    assert Superkey(Meta("cols")).check_with_context(basic_df, {"cols": ["a"]}) is None
