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
    List,
    Mapping,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from pandas.core.dtypes import common

from signpost import utils

S = TypeVar("S")
T = TypeVar("T")
ColsType = utils.Wrappable[Hashable]
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

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        result = self.check(df)
        if result is not None:
            raise DataFrameTypeError(result)
        return df

    def __and__(self, other: object) -> "And":
        if not isinstance(other, (Property, ContextProperty)):
            raise TypeError(f"And operation not supported between {self} and {other}")
        return And(self, other)

    def __or__(self, other: object) -> "Or":
        if not isinstance(other, (Property, ContextProperty)):
            raise TypeError(f"Or operation not supported between {self} and {other}")
        return Or(self, other)


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

    def __and__(self, other: object) -> "And":
        if not isinstance(other, (Property, ContextProperty)):
            raise TypeError(f"And operation not supported between {self} and {other}")
        return And(self, other)

    def __or__(self, other: object) -> "Or":
        if not isinstance(other, (Property, ContextProperty)):
            raise TypeError(f"Or operation not supported between {self} and {other}")
        return Or(self, other)


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


class QualifierEvaluator:
    def __init__(
        self, qualifier: Qualifier, reference: pd.DataFrame, comparison: pd.DataFrame
    ):
        self.qualifier = qualifier
        self.reference = reference
        self.comparison = comparison

    def _get_overlap_parts(self) -> Set[str]:
        indicator_col = str(uuid.uuid5(uuid.NAMESPACE_URL, "signpost"))
        result = (
            pd.merge(
                self.reference, self.comparison, how="outer", indicator=indicator_col
            )
            .loc[:, indicator_col]
            .drop_duplicates()
            .pipe(set)
        )
        return cast(Set[str], result)

    def _has_all(self) -> bool:
        return "right_only" not in self._get_overlap_parts()

    def _has_any(self) -> bool:
        return "both" in self._get_overlap_parts()

    def _has_none(self) -> bool:
        return not self._has_any()

    def _has_just(self) -> bool:
        return self._get_overlap_parts() == {"both"}

    def eval(self) -> bool:
        return {
            Qualifier.ALL: self._has_all,
            Qualifier.ANY: self._has_any,
            Qualifier.NONE: self._has_none,
            Qualifier.JUST: self._has_just,
        }[self.qualifier]()


class Cols(ContextProperty):
    class Checker(Property):
        def __init__(self, qualifier: Union[str, Qualifier], cols: ColsType):
            self.qualifier = Qualifier(qualifier)
            self.cols = utils.wrap(cols)

        def _raw_check(self, df: pd.DataFrame) -> bool:
            return QualifierEvaluator(
                self.qualifier,
                reference=pd.DataFrame({"col": list(df)}),
                comparison=pd.DataFrame({"col": self.cols}),
            ).eval()

        def check(self, df: pd.DataFrame) -> Optional[str]:
            if not self._raw_check(df):
                return f"Expected {self.qualifier.value} columns:\n{self.cols}\nFound:\n{list(df)}"
            return None

    def __init__(
        self, qualifier: MetaVar[Union[str, Qualifier]], cols: MetaVar[ColsType]
    ):
        self.qualifier = Param(qualifier, Qualifier)
        self.cols = Param(cols, lambda x: utils.wrap(x))

    def check_with_context(
        self, df: pd.DataFrame, context: Dict[str, Any]
    ) -> Optional[str]:
        return self.Checker(
            self.qualifier.get(context), cols=self.cols.get(context)
        ).check(df)


class Schema(ContextProperty):
    class Checker(Property):
        def __init__(self, qualifier: Union[str, Qualifier], schema: SchemaType):
            self.qualifier = Qualifier(qualifier)
            self.schema = schema

        @staticmethod
        def make_types_df(schema: SchemaType) -> pd.DataFrame:
            return pd.DataFrame(schema.items(), columns=["col", "dtype"]).assign(
                dtype=lambda d: d["dtype"].map(common.pandas_dtype)
            )

        def _raw_check(self, df: pd.DataFrame) -> bool:
            return QualifierEvaluator(
                self.qualifier,
                reference=self.make_types_df(cast(SchemaType, df.dtypes.to_dict())),
                comparison=self.make_types_df(self.schema),
            ).eval()

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
    class Checker(Property):
        def __init__(
            self,
            qualifier: Union[str, Qualifier],
            values: Dict[Hashable, Collection[Any]],
        ):
            self.qualifier = Qualifier(qualifier)
            self.values = values

        def check(self, df: pd.DataFrame) -> Optional[str]:
            col_check = Cols.Checker("all", list(self.values)).check(df)
            if col_check is not None:
                return col_check

            evaluator = QualifierEvaluator(
                self.qualifier,
                reference=df.loc[:, self.values].drop_duplicates(),
                comparison=(
                    pd.DataFrame(self.values)
                    .astype(df.dtypes[self.values])
                    .drop_duplicates()
                ),
            )
            if not evaluator.eval():
                return (
                    f"Expected {self.qualifier.value} values:\n{evaluator.comparison}\n"
                    f"Found:\n{evaluator.reference}"
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
            self.cols = utils.wrap(cols)
            self.over = utils.wrap(over) if over is not None else None

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
        self.cols = Param(cols, lambda x: utils.wrap(x))
        self.over = Param(over, lambda x: utils.wrap(x) if x is not None else None)

    def check_with_context(
        self, df: pd.DataFrame, context: Dict[str, Any]
    ) -> Optional[str]:
        return self.Checker(self.cols.get(context), self.over.get(context)).check(df)


class Assume(ContextProperty):
    """
    Wraps a ContextProperty or Property to treat it as always true.

    In order to document a property that we know to be true but may be
    expensive to compute, we can use the Assume property
    """

    def __init__(self, inner: Union[ContextProperty, Property]):
        self.inner = inner

    def check_with_context(
        self, df: pd.DataFrame, context: Dict[str, Any]
    ) -> Optional[str]:
        return None


class Notna(ContextProperty):
    class Checker(Property):
        def __init__(self, qualifier: Union[str, Qualifier], cols: ColsType):
            self.qualifier = Qualifier(qualifier)
            self.cols = utils.wrap(cols)

        def _get_error_string(
            self,
            df: pd.DataFrame,
            notna: pd.DataFrame,
            notna_cols: Collection[Hashable],
        ) -> str:
            if self.qualifier == Qualifier.NONE:
                # this is a double negative sort of case
                return (
                    f"Expected NA values in these columns:\n{self.cols}\n"
                    f"But only these columns contain NA:\n{list(set(df) - set(notna_cols))}"
                )
            else:
                human_readable = {
                    Qualifier.ANY: "at least one of",
                    Qualifier.ALL: "all of",
                    Qualifier.JUST: "just",
                }[self.qualifier]
                return (
                    f"Expected no NA values in {human_readable} these columns:\n{self.cols}\n"
                    f"But only these columns are not NA:\n{notna_cols}\n"
                    "These rows contain NA values:\n"
                    f"{df.loc[~notna.loc[:, self.cols].all(axis='columns'), :]}"
                )

        def check(self, df: pd.DataFrame) -> Optional[str]:
            col_check = Cols.Checker("all", self.cols).check(df)
            if col_check is not None:
                return col_check

            notna = df.notna()
            evaluator = QualifierEvaluator(
                self.qualifier,
                reference=pd.DataFrame({"col": notna.all().loc[lambda s: s].index}),
                comparison=pd.DataFrame({"col": self.cols}),
            )
            return (
                self._get_error_string(df, notna, evaluator.reference["col"].tolist())
                if not evaluator.eval()
                else None
            )

    def __init__(
        self, qualifier: MetaVar[Union[str, Qualifier]], cols: MetaVar[ColsType]
    ):
        self.qualifier = Param(qualifier, Qualifier)
        self.cols = Param(cols, lambda x: utils.wrap(x))

    def check_with_context(
        self, df: pd.DataFrame, context: Dict[str, Any]
    ) -> Optional[str]:
        return self.Checker(self.qualifier.get(context), self.cols.get(context)).check(
            df
        )


class MergeResult(ContextProperty):
    class Checker(Property):
        def __init__(
            self,
            merge_results: utils.Wrappable[str],
            indicator_col: Hashable = "_merge",
        ):
            self.results: List[str] = utils.wrap(merge_results)
            self.col = indicator_col

        def check(self, df: pd.DataFrame) -> Optional[str]:
            return Values.Checker(Qualifier.JUST, {self.col: self.results}).check(df)

    def __init__(
        self,
        merge_results: MetaVar[utils.Wrappable[str]],
        indicator_col: MetaVar[Hashable] = "_merge",
    ):
        self.results: Param[utils.Wrappable[str], List[str]] = Param(
            merge_results, lambda x: utils.wrap(x)
        )
        self.col = Param(indicator_col, lambda x: x)

    def check_with_context(
        self, df: pd.DataFrame, context: Dict[str, Any]
    ) -> Optional[str]:
        return self.Checker(
            merge_results=self.results.get(context), indicator_col=self.col.get(context)
        ).check(df)


class Bounded(ContextProperty):
    @unique
    class Closed(Enum):
        BOTH = "both"
        LEFT = "left"
        RIGHT = "right"
        NEITHER = "neither"

        @property
        def closed_left(self) -> bool:
            return self in {self.BOTH, self.LEFT}

        @property
        def closed_right(self) -> bool:
            return self in {self.BOTH, self.RIGHT}

    class Checker(Property):
        def __init__(
            self,
            cols: ColsType,
            lower: Optional[Any],
            upper: Optional[Any],
            closed: Union[str, "Bounded.Closed"],
        ):
            self.cols = utils.wrap(cols)
            self.lower = lower
            self.upper = upper
            if (
                self.lower is not None
                and self.upper is not None
                and self.upper < self.lower
            ):
                raise ValueError(
                    f"The upper bound {self.upper} must be greater than the lower bound {self.lower}"
                )
            self.closed = Bounded.Closed(closed)
            if self.lower is None and self.closed.closed_left:
                raise ValueError("An interval cannot be left-closed and left-unbounded")
            if self.upper is None and self.closed.closed_right:
                raise ValueError(
                    "An interval cannot be right-closed and right-unbounded"
                )

        def _format_bounds(self) -> str:
            lower = "-inf" if self.lower is None else str(self.lower)
            upper = "inf" if self.upper is None else str(self.upper)
            lower_bracket = "[" if self.closed.closed_left else "("
            upper_bracket = "]" if self.closed.closed_right else ")"
            return f"{lower_bracket}{lower}, {upper}{upper_bracket}"

        def _check_lower(self, value_df: pd.DataFrame) -> pd.Series:
            if self.lower is not None:
                # noinspection PyUnresolvedReferences
                return (
                    value_df >= self.lower
                    if self.closed.closed_left
                    else value_df > self.lower
                ).all(axis="columns")
            else:
                return pd.Series(np.full(len(value_df), True))

        def _check_upper(self, value_df: pd.DataFrame) -> pd.Series:
            if self.upper is not None:
                # noinspection PyUnresolvedReferences
                return (
                    value_df <= self.upper
                    if self.closed.closed_right
                    else value_df < self.upper
                ).all(axis="columns")
            else:
                return pd.Series(np.full(len(value_df), True))

        def check(self, df: pd.DataFrame) -> Optional[str]:
            col_check = Cols.Checker("all", self.cols).check(df)
            if col_check is not None:
                return col_check

            value_df = df.loc[:, self.cols]
            valid = self._check_lower(value_df) & self._check_upper(value_df)
            if not valid.all():
                return (
                    f"Some observations for {self.cols} fall outside the bounds {self._format_bounds()}:\n"
                    f"{df.loc[~valid, :]}"
                )
            else:
                return None

    def __init__(
        self,
        cols: MetaVar[ColsType],
        lower: MetaVar[Optional[Any]],
        upper: MetaVar[Optional[Any]],
        closed: MetaVar["Bounded.Closed"],
    ):
        self.cols = Param(cols, lambda x: utils.wrap(x))
        self.lower: Param[Optional[Any], Optional[Any]] = Param(lower, lambda x: x)
        self.upper: Param[Optional[Any], Optional[Any]] = Param(upper, lambda x: x)
        self.closed = Param(closed, lambda x: Bounded.Closed(x))

    def check_with_context(
        self, df: pd.DataFrame, context: Dict[str, Any]
    ) -> Optional[str]:
        return self.Checker(
            cols=self.cols.get(context),
            lower=self.lower.get(context),
            upper=self.upper.get(context),
            closed=self.closed.get(context),
        ).check(df)


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
