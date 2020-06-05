signpost
========

This is a simple library for annotating and enforcing properties of
pandas DataFrames at runtime. By showing which columns and types
are expected before execution of a function begins, we can catch errors
earlier with a message that makes sense and also document the inputs and
outputs of functions more concisely.


Example Usage
-------------

Here is an example of standard usage to decorate a fairly strange function.
Note that any valid pandas index value can be used, including numbers. We
can combine ``Property``'s together using ``And`` and ``Or`` if desired
as well as qualify them using "all", "any", "just", or "none".

.. code-block:: python

    from signpost import Cols, Schema, Values, Superkey, And, df_args, df_return

    @df_args(
        And(Cols("all", ["thing_1", 2]), Superkey(["thing_1"], over=[2])),
        other=Schema("just", {"thing_2": int, "thing_3": "string"})
    )
    @df_return(
        None,
        And(
            Cols("all", ["thing_1", "thing_2", "thing_3"]),
            Values("any", {"thing_1": [1, 2], "thing_3": ["foo", "bar"]}),
            Values("none", {"thing_1": [3]}),
        )
    )
    def do_a_thing(df: pd.DataFrame, other: pd.DataFrame) -> (int, pd.DataFrame):
        ...

However, there are times when the particular properties of a data frame depend on other
inputs to a function. For example, a function may take a list of columns to subset
by or a set of values to query with. This behavior is somewhat analogous to a function
taking a ``List[T]`` and a parameter of type ``T`` – we are essentially making the data
frame generic over a parameter specified by the caller. In these cases, we can
use the ``Meta`` constructor, which is constructed with a string of Python code.
The code is then evaluated with the environment of the function.
For example, we can implement a checked "project" function
(analogous to ``SELECT DISTINCT`` in SQL) as follows:

.. code-block:: python

    from signpost import df_args, df_return, Cols, Meta

    @df_args(Cols("all", Meta("cols")))
    @df_return(Cols("just", Meta("cols")))
    def project(df: pd.DataFrame, cols: List[str]):
        return df.loc[:, cols].drop_duplicates()

Since the expressions passed to these meta properties can be arbitrary Python strings,
we can express some fairly powerful logic using relatively little code. Note that
since pandas DataFrames are dict-like, we can treat them as sequences of column names.

.. code-block:: python

    from signpost import df_args, df_return, Cols, Meta

    @df_args(left=Cols("any", Meta("right")), right=Cols("any", Meta("left")))
    @df_return(Cols("just", Meta("set(left) | set(other)"))
    def merge(left, right):
        return pd.merge(left, right)

Extending signpost
------------------
There are a couple of ways to extend signpost. The first is using the ``Function`` property.
It simply accepts a function that takes a pandas DataFrame and a context dictionary and returns
a ``Optional[str]``.

.. code-block:: python

    from signpost import df_args, Function

    @df_args(Function(lambda df, context: "bad" if df.empty else None))
    def do_another_thing(df: pd.DataFrame):
        ...

It is also possible to create new ``Property``'s simply by implementing the ``Property``
or ``ContextProperty`` interface found in ``signpost.properties``.


TODO
----
There are a couple of improvements to be made, namely

1. **Ergonomics.** Assume bare types to be single-element lists.

2. **Documentation.**