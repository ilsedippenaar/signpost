import uuid
from enum import Enum, unique
from types import CodeType
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    Hashable,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import pandas as pd
from pandas.core.dtypes import common

S = TypeVar("S")
T = TypeVar("T")
ColsType = Collection[Hashable]
SchemaType = Mapping[Hashable, Type[Any]]


class DataFrameTypeError(TypeError):
    pass


@unique
class Qualifier(Enum):
    ALL = "all"
    ANY = "any"
    NONE = "none"
    JUST = "just"


class Property:
    def check(self, df: pd.DataFrame) -> Optional[str]:
        raise NotImplementedError


class ContextProperty:
    """
    This mixin should be treated as a builder type for `Property`'s.
    The additional `context` argument should be used to instantiate arguments
    at runtime if need be using the `Param` helper class.
    """

    def check_with_context(
        self, df: pd.DataFrame, context: Dict[str, Any]
    ) -> Optional[str]:
        raise NotImplementedError


class Meta(Generic[T]):
    def __init__(self, expr: str):
        self.expr = cast(CodeType, compile(expr, "<string>", "eval"))

    def eval(self, context: Dict[str, Any]) -> T:
        return cast(T, eval(self.expr, globals(), context))


MetaVar = Union[T, Meta[T]]


class Param(Generic[S, T]):
    """
    Helper class for constructing parameters that may be calculated at
    runtime by `Meta` from a given context.
    """

    def __init__(self, val: MetaVar[S], process: Callable[[S], T]):
        self.process = process
        self.val: Union[Meta[S], T] = val if isinstance(val, Meta) else self.process(
            val
        )

    def get(self, context: Dict[str, Any]) -> T:
        if isinstance(self.val, Meta):
            return self.process(self.val.eval(context))
        return self.val


class QualifierMixin:
    def _get_compare_dfs(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    def _eval_condition(self, df: pd.DataFrame) -> Set[str]:
        left, right = self._get_compare_dfs(df)
        indicator_col = str(uuid.uuid5(uuid.NAMESPACE_URL, "signpost"))
        result = (
            pd.merge(left, right, how="outer", indicator=indicator_col)
            .loc[:, indicator_col]
            .drop_duplicates()
            .pipe(set)
        )
        return cast(Set[str], result)

    def _has_all(self, df: pd.DataFrame) -> bool:
        return "right_only" not in self._eval_condition(df)

    def _has_any(self, df: pd.DataFrame) -> bool:
        return "both" in self._eval_condition(df)

    def _has_none(self, df: pd.DataFrame) -> bool:
        return not self._has_any(df)

    def _has_just(self, df: pd.DataFrame) -> bool:
        return self._eval_condition(df) == {"both"}

    def _get_qualified_eval_func(
        self, qualifier: Qualifier
    ) -> Callable[[pd.DataFrame], bool]:
        return {
            Qualifier.ALL: self._has_all,
            Qualifier.ANY: self._has_any,
            Qualifier.NONE: self._has_none,
            Qualifier.JUST: self._has_just,
        }[qualifier]


class Cols(ContextProperty):
    class Checker(Property, QualifierMixin):
        def __init__(self, qualifier: Union[str, Qualifier], cols: ColsType):
            self.qualifier = Qualifier(qualifier)
            self.cols = list(cols)

        def _get_compare_dfs(
            self, df: pd.DataFrame
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
            return pd.DataFrame({"col": list(df)}), pd.DataFrame({"col": self.cols})

        def _raw_check(self, df: pd.DataFrame) -> bool:
            return self._get_qualified_eval_func(self.qualifier)(df)

        def check(self, df: pd.DataFrame) -> Optional[str]:
            if not self._raw_check(df):
                return f"Expected {self.qualifier.value} columns:\n{self.cols}\nFound:\n{list(df)}"
            return None

    def __init__(
        self, qualifier: MetaVar[Union[str, Qualifier]], cols: MetaVar[ColsType]
    ):
        self.qualifier = Param(qualifier, Qualifier)
        self.cols = Param(cols, lambda x: list(x))

    def check_with_context(
        self, df: pd.DataFrame, context: Dict[str, Any]
    ) -> Optional[str]:
        return self.Checker(
            self.qualifier.get(context), cols=self.cols.get(context)
        ).check(df)


class Schema(ContextProperty):
    class Checker(Property, QualifierMixin):
        def __init__(self, qualifier: Union[str, Qualifier], schema: SchemaType):
            self.qualifier = Qualifier(qualifier)
            self.schema = schema

        @staticmethod
        def make_types_df(schema: SchemaType) -> pd.DataFrame:
            return pd.DataFrame(schema.items(), columns=["col", "dtype"]).assign(
                dtype=lambda d: d["dtype"].map(common.pandas_dtype)
            )

        def _get_compare_dfs(
            self, df: pd.DataFrame
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
            return (
                self.make_types_df(cast(SchemaType, df.dtypes.to_dict())),
                self.make_types_df(self.schema),
            )

        def _raw_check(self, df: pd.DataFrame) -> bool:
            return self._get_qualified_eval_func(self.qualifier)(df)

        def check(self, df: pd.DataFrame) -> Optional[str]:
            if not self._raw_check(df):
                return f"Expected {self.qualifier.value} types:\n{self.schema}\nFound:\n{df.dtypes.to_dict()}"
            return None

    def __init__(
        self, qualifier: MetaVar[Union[str, Qualifier]], schema: MetaVar[SchemaType]
    ):
        self.qualifier = Param(qualifier, Qualifier)
        self.schema = Param(schema, lambda x: x)

    def check_with_context(
        self, df: pd.DataFrame, context: Dict[str, Any]
    ) -> Optional[str]:
        return self.Checker(
            self.qualifier.get(context), self.schema.get(context)
        ).check(df)


class Values(ContextProperty):
    class Checker(Property, QualifierMixin):
        def __init__(
            self,
            qualifier: Union[str, Qualifier],
            values: Dict[Hashable, Collection[Any]],
        ):
            self.qualifier = Qualifier(qualifier)
            self.values = values

        def _get_compare_dfs(
            self, df: pd.DataFrame
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
            return (
                df.loc[:, self.values].drop_duplicates(),
                pd.DataFrame(self.values)
                .astype(df.dtypes[self.values])
                .drop_duplicates(),
            )

        def _raw_check(self, df: pd.DataFrame) -> bool:
            return self._get_qualified_eval_func(self.qualifier)(df)

        def check(self, df: pd.DataFrame) -> Optional[str]:
            col_check = Cols.Checker("all", list(self.values)).check(df)
            if col_check is not None:
                return col_check

            if not self._raw_check(df):
                return (
                    f"Expected {self.qualifier.value} values:\n{self.values}\n"
                    f"Found:\n{df[self.values].drop_duplicates()}"
                )
            return None

    def __init__(
        self,
        qualifier: MetaVar[Union[str, Qualifier]],
        values: MetaVar[Dict[Hashable, Collection[Any]]],
    ):
        self.qualifier = Param(qualifier, Qualifier)
        self.values = Param(values, lambda x: x)

    def check_with_context(
        self, df: pd.DataFrame, context: Dict[str, Any]
    ) -> Optional[str]:
        return self.Checker(
            self.qualifier.get(context), self.values.get(context)
        ).check(df)


class Superkey(ContextProperty):
    class Checker(Property):
        def __init__(self, cols: ColsType, over: Optional[ColsType] = None):
            self.cols = list(cols)
            self.over = list(over) if over is not None else None

        def _get_duplicated(self, df: pd.DataFrame) -> pd.DataFrame:
            if not self.cols:
                return df
            elif self.over is not None:
                df = df.loc[:, self.cols + self.over].drop_duplicates()
            return df.loc[lambda d: d.duplicated(self.cols, keep=False)]

        def check(self, df: pd.DataFrame) -> Optional[str]:
            check_cols = self.cols + (self.over or [])
            col_check = Cols.Checker("all", check_cols).check(df)
            if col_check is not None:
                return col_check
            else:
                df = self._get_duplicated(df)
                if df.empty:
                    return None
                else:
                    return (
                        f"Columns {self.cols} do not form a superkey "
                        + (f"over {self.over} " if self.over is not None else "")
                        + f"for the data:\n{df}"
                    )

    def __init__(
        self, cols: MetaVar[ColsType], over: MetaVar[Optional[ColsType]] = None
    ):
        self.cols = Param(cols, lambda x: list(x))
        self.over = Param(over, lambda x: list(x) if x is not None else None)

    def check_with_context(
        self, df: pd.DataFrame, context: Dict[str, Any]
    ) -> Optional[str]:
        return self.Checker(self.cols.get(context), self.over.get(context)).check(df)


class Function(ContextProperty):
    def __init__(
        self, function: Callable[[pd.DataFrame, Dict[str, Any]], Optional[str]]
    ):
        self.function = function

    def check_with_context(
        self, df: pd.DataFrame, context: Dict[str, Any]
    ) -> Optional[str]:
        return self.function(df, context)


class And(ContextProperty):
    def __init__(self, *properties: Union[ContextProperty, Property]):
        self.properties = properties

    def check_with_context(
        self, df: pd.DataFrame, context: Dict[str, Any]
    ) -> Optional[str]:
        for prop in self.properties:
            result = (
                prop.check_with_context(df, context)
                if isinstance(prop, ContextProperty)
                else prop.check(df)
            )
            if result is not None:
                return result
        return None


class Or(ContextProperty):
    def __init__(self, *properties: Union[ContextProperty, Property]):
        self.properties = properties

    def check_with_context(
        self, df: pd.DataFrame, context: Dict[str, Any]
    ) -> Optional[str]:
        results = []
        for prop in self.properties:
            current = (
                prop.check_with_context(df, context)
                if isinstance(prop, ContextProperty)
                else prop.check(df)
            )
            if current is None:
                return None
            else:
                results.append(current)

        return "\n".join(results)
