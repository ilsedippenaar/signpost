signpost
========

|pypi-version| |pypi-python-versions| |build-status| |coverage|

This is a simple library for annotating and enforcing properties of
pandas DataFrames at runtime. By showing which columns and types
are expected before execution of a function begins, we can catch errors
earlier with a message that makes sense and also document the inputs and
outputs of functions more concisely.

This project is related to a number of others that all share similar goals.
While perhaps stumbling straight headlong the Lisp curse, signpost is yet
another pandas validation library. Here is a list of other similar projects:

* `Bulwark <https://github.com/zaxr/bulwark>`_
* `pandera <https://github.com/pandera-dev/pandera>`_
* `Table Enforcer <https://github.com/xguse/table_enforcer>`_
* `pandas-validation <https://github.com/jmenglund/pandas-validation>`_
* `PandasSchema <https://github.com/TMiguelT/PandasSchema>`_
* `Opulent-Pandas <https://github.com/danielvdende/opulent-pandas>`_

So why reinvent the wheel? Signpost offers a few advantages:

#. Support for delayed evaluation of property inputs through the use of ``Meta``.
   This technique works especially well in settings where class variables may hold information
   about the data being operated on by the class.
#. Qualifiers allow for richer and more flexible descriptions of data
#. Straightforward approach to function decorators that uses the same logic as Python itself
   to match properties to DataFrames
#. Strict Python type checking via mypy


Example Usage
-------------

Here is an example of standard usage to decorate a fairly strange function.
Note that any valid pandas index value can be used, including numbers. We
can combine ``Property``'s together using ``And`` and ``Or`` if desired
as well as qualify them using "all", "any", "just", or "none".

.. code-block:: python

    from signpost import Cols, Schema, Values, Superkey, And, df_args, df_return

    @df_args(
        Cols("all", ["thing_1", 2]) & Superkey("thing_1", over=2),
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
    def do_a_thing(df: pd.DataFrame, other: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
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
    def project(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
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

Other use cases
^^^^^^^^^^^^^^^
For usage inside scripts, it is useful to use the ``Checker`` inner class for various properties.
For example,

.. code-block:: python

    from signpost import Cols, Values

    df = pd.read_csv("my_file.csv")
    df = Cols.Checker("just", ["col_a", "col_b"]).validate(df)
    df = Values.Checker("all", {"col_a": [1, 2], "col_b": [1, 1]}).validate(df)

When combined with ``pd.DataFrame.pipe``, ``validate`` can provide expressive sanity checking.
If you would like more custom handling, you can use the ``check`` method as follows:

.. code-block:: python

    from signpost import Cols

    df = ...
    error: Optional[str] = Cols.Checker("just", ["col_a", "col_b"]).check(df)
    if error is not None:
        print(error)
        # more handling
        ...


List of Properties
------------------

* Cols: checks that the specified columns are in the data
* Schema: checks whether the specified column / data type pairs match the data
* Values: enforces which values (and combinations of values) need to be present in the data
* Superkey: checks that the specified columns uniquely identify the data
* Notna: enforces that the specified columns do not contain NA / missing values
* MergeResult: checks whether a merge was a inner, left, or right join
* Bounded: enforces that the values in the specified columns fall between two (potentially unbounded) values

Special properties
^^^^^^^^^^^^^^^^^^
* Function: wraps a bare function into a property, useful for quick checks
* And: combines two properties into a new property that checks each in turn, stopping if an error is found
* Or: combines two properties into a new property that checks each in turn, stopping once a property succeeds
* Assume: wraps a property to always be true, useful for documenting a property without doing unnecessary computation


Installation
------------

Installation is easy! Just type:

.. code-block:: console

    pip install signpost

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


.. |pypi-version| image:: https://img.shields.io/pypi/v/signpost
    :alt: PyPI
    :target: https://pypi.org/project/signpost

.. |pypi-python-versions| image:: https://img.shields.io/pypi/pyversions/signpost
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/signpost

.. |build-status| image:: https://travis-ci.com/ilsedippenaar/signpost.svg?branch=main
    :alt: Build Status
    :target: https://travis-ci.com/ilsedippenaar/signpost

.. |coverage| image:: https://codecov.io/gh/ilsedippenaar/signpost/branch/main/graph/badge.svg
    :alt: Code Coverage
    :target: https://codecov.io/gh/ilsedippenaar/signpost
