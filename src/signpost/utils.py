from typing import Collection, List, TypeVar, Union

from pandas.core.common import index_labels_to_array

T = TypeVar("T")
Wrappable = Union[T, Collection[T]]


def wrap(labels: Wrappable[T]) -> List[T]:
    # noinspection PyTypeChecker
    return list(index_labels_to_array(labels))
