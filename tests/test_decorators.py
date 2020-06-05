from typing import Any, Callable, Dict, Hashable, Iterable, List, Tuple, Union

import pandas as pd
import pytest

from signpost import Cols, Meta, df_args, df_return
from signpost import properties as props


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(columns=["a", "b"])


def get_other() -> pd.DataFrame:
    return pd.DataFrame(columns=["a", "c"])


def basic_func(df: pd.DataFrame) -> pd.DataFrame:
    return df


def varied_args_func(
    _: int = 0, df: pd.DataFrame = None, *args: Any, **kwargs: Any
) -> pd.DataFrame:
    return df


def varied_return_func(df: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
    return False, df


def merge_func(df: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    return df.merge(other)


def project_func(df: pd.DataFrame, cols: List[Hashable]) -> pd.DataFrame:
    return df.loc[:, cols].drop_duplicates()


@pytest.mark.parametrize(
    "prop_map, func, args, kwargs",
    [
        ({"df": Cols("just", ["a", "b"])}, basic_func, [], {}),
        ({"df": Cols("just", ["a", "b"])}, varied_args_func, [], {"_": 5}),
        ({"df": Cols("just", ["a", "b"])}, varied_args_func, [1], {}),
        (
            {"df": Cols("just", ["a", "b"]), "other": Cols("just", ["a", "c"])},
            merge_func,
            [],
            {"other": get_other()},
        ),
        (
            {"df": Cols("any", Meta("other")), "other": Cols("any", Meta("df"))},
            merge_func,
            [],
            {"other": get_other()},
        ),
        pytest.param(
            {"df": Cols("just", ["a"])},
            basic_func,
            [],
            {},
            marks=pytest.mark.xfail(raises=props.DataFrameTypeError, strict=True),
        ),
        pytest.param(
            {"other_df": Cols("just", ["x"])},
            basic_func,
            [],
            {},
            marks=pytest.mark.xfail(raises=TypeError, strict=True),
        ),
    ],
)
def test_df_args(
    df: pd.DataFrame,
    prop_map: Dict[str, Union[props.Property, props.ContextProperty]],
    func: Callable[[pd.DataFrame], Any],
    args: Iterable[Any],
    kwargs: Dict[str, Any],
) -> None:
    decorator = df_args(**prop_map)
    decorator(func)(df=df, *args, **kwargs)


@pytest.mark.parametrize(
    "properties, func, args, kwargs",
    [
        ([Cols("just", ["a", "b"])], basic_func, [], {}),
        ([Cols("just", ["a", "b"])], varied_args_func, [1], {}),
        ([None, Cols("just", ["a", "b"])], varied_return_func, [], {}),
        ([Cols("just", Meta("cols"))], project_func, [], {"cols": ["a", "b"]}),
        (
            [Cols("just", Meta("set(df) | set(other)"))],
            merge_func,
            [],
            {"other": get_other()},
        ),
        pytest.param(
            [Cols("none", Meta("cols"))],
            project_func,
            [],
            {"cols": ["a"]},
            marks=pytest.mark.xfail(raises=props.DataFrameTypeError, strict=True),
        ),
    ],
)
def test_df_return(
    df: pd.DataFrame,
    properties: List[Union[props.Property, props.ContextProperty]],
    func: Callable[[pd.DataFrame], Any],
    args: Iterable[Any],
    kwargs: Dict[str, Any],
) -> None:
    decorator = df_return(*properties)
    decorator(func)(df=df, *args, **kwargs)
