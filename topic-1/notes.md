# Topic 1: Pandas Data Analysis Notes

These notes summarize the main functions and approaches from the mlcourse.ai pandas lesson.

## Basic Setup

```python
import numpy as np
import pandas as pd

pd.set_option("display.precision", 2)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)
```

- `import pandas as pd`: loads pandas for working with tables.
- `import numpy as np`: loads NumPy, often used together with pandas.
- `pd.set_option(...)`: changes how pandas displays tables in notebooks.

## Loading And Inspecting Data

```python
df = pd.read_csv("file.csv")
df.head()
df.shape
df.columns
df.info()
df.describe()
```

- `pd.read_csv(...)`: reads a CSV file into a DataFrame.
- `df.head()`: shows the first 5 rows.
- `df.shape`: returns `(number_of_rows, number_of_columns)`.
- `df.columns`: shows all column names.
- `df.info()`: shows column types, missing values, and memory usage.
- `df.describe()`: gives statistics for numeric columns: count, mean, std, min, quartiles, max.

For categorical columns:

```python
df.describe(include=["object", "bool"])
```

## Counting Values

```python
df["column"].value_counts()
df["column"].value_counts(normalize=True)
```

- `value_counts()`: counts how often each value appears.
- `normalize=True`: returns proportions instead of raw counts.

Example: percentage of German citizens:

```python
(data["native-country"] == "Germany").mean() * 100
```

Boolean values act like numbers: `True = 1`, `False = 0`, so `.mean()` gives the proportion.

## Changing Types

```python
df["Churn"] = df["Churn"].astype("int64")
```

- `astype(...)`: converts a column to another data type.
- Useful when booleans should become `0` and `1`.

## Sorting

```python
df.sort_values(by="Total day charge", ascending=False).head()
```

- `sort_values(...)`: sorts rows by column values.
- `ascending=False`: sorts from largest to smallest.

Sort by multiple columns:

```python
df.sort_values(
    by=["Churn", "Total day charge"],
    ascending=[True, False]
)
```

## Selecting Columns

```python
df["age"]
df[["age", "sex", "salary"]]
```

- `df["age"]`: selects one column as a Series.
- `df[["age", "sex"]]`: selects multiple columns as a DataFrame.

## Boolean Filtering

```python
df[df["salary"] == ">50K"]
df[(df["salary"] == ">50K") & (df["sex"] == "Female")]
```

- Boolean filtering keeps only rows where the condition is `True`.
- Use `&` for AND.
- Use `|` for OR.
- Wrap each condition in parentheses.

Example: average age of women:

```python
data.loc[data["sex"] == "Female", "age"].mean()
```

## `loc` And `iloc`

```python
df.loc[rows, columns]
df.iloc[row_numbers, column_numbers]
```

- `loc`: selects by labels or boolean conditions.
- `iloc`: selects by integer positions.

Examples:

```python
df.loc[0:5, "State":"Area code"]
df.iloc[0:5, 0:3]
```

Important difference:

- `loc[0:5]` includes label `5`.
- `iloc[0:5]` stops before position `5`.

Useful pattern:

```python
data.loc[data["salary"] == ">50K", "education"]
```

This means: select rows where salary is `>50K`, then return only the `education` column.

## Aggregation

```python
df["age"].mean()
df["age"].std()
df["age"].max()
df["age"].min()
```

Common aggregation functions:

- `mean()`: average
- `std()`: standard deviation
- `max()`: largest value
- `min()`: smallest value
- `sum()`: total
- `count()`: number of non-missing values

## Grouping

```python
df.groupby("salary")["age"].mean()
```

- `groupby(...)`: splits data into groups.
- Then you choose a column and apply a function.

Example: mean and standard deviation of age by salary:

```python
data.groupby("salary")["age"].agg(["mean", "std"])
```

General pattern:

```python
df.groupby(grouping_columns)[columns_to_show].function()
```

Multiple statistics:

```python
df.groupby("Churn")[["Total day minutes", "Total eve minutes"]].agg(
    ["mean", "std", "min", "max"]
)
```

## Applying Functions

```python
df.apply(np.max)
```

- `apply(...)`: applies a function to columns or rows.
- By default, it works column by column.

With a custom function:

```python
df[df["State"].apply(lambda state: state[0] == "W")]
```

- `lambda`: a small inline function.
- This example selects states starting with `"W"`.

## Mapping And Replacing Values

```python
d = {"No": False, "Yes": True}

df["International plan"] = df["International plan"].map(d)
df = df.replace({"Voice mail plan": d})
```

- `map(...)`: replaces values in one column using a dictionary.
- `replace(...)`: replaces values in a column or whole DataFrame.

Difference:

- `map(...)` turns values missing from the dictionary into `NaN`.
- `replace(...)` leaves unknown values unchanged.

## Crosstab

```python
pd.crosstab(df["Churn"], df["International plan"])
pd.crosstab(df["Churn"], df["International plan"], margins=True)
```

- `pd.crosstab(...)`: creates a frequency table for two variables.
- `margins=True`: adds row and column totals.

Normalize to proportions:

```python
pd.crosstab(df["Churn"], df["Voice mail plan"], normalize=True)
```

## Pivot Tables

```python
df.pivot_table(
    ["Total day calls", "Total eve calls", "Total night calls"],
    ["Area code"],
    aggfunc="mean",
)
```

- `pivot_table(...)`: groups data like an Excel pivot table.
- `values`: columns to calculate statistics for.
- `index`: columns to group by.
- `aggfunc`: statistic to calculate, such as `"mean"` or `"sum"`.

## Adding Columns

```python
df["Total charge"] = (
    df["Total day charge"]
    + df["Total eve charge"]
    + df["Total night charge"]
    + df["Total intl charge"]
)
```

- You can create new columns from existing columns.
- Pandas does the calculation row by row.

Using `insert`:

```python
df.insert(loc=len(df.columns), column="Total calls", value=total_calls)
```

- `insert(...)`: inserts a column at a specific position.

## Dropping Columns Or Rows

```python
df.drop(["Total charge", "Total calls"], axis=1, inplace=True)
df.drop([1, 2])
```

- `axis=1`: drop columns.
- `axis=0`: drop rows. This is the default.
- `inplace=True`: changes the original DataFrame.
- `inplace=False`: returns a new DataFrame.

## Checking If All Rows Match A Condition

Example: do all people earning `>50K` have high education?

```python
high_education = [
    "Bachelors",
    "Prof-school",
    "Assoc-acdm",
    "Assoc-voc",
    "Masters",
    "Doctorate",
]

data.loc[data["salary"] == ">50K", "education"].isin(high_education).all()
```

- `isin(list)`: checks whether each value is in a list.
- `all()`: returns `True` only if all values are `True`.

To find counterexamples:

```python
data.loc[
    (data["salary"] == ">50K") & (~data["education"].isin(high_education)),
    "education"
].unique()
```

- `~`: means NOT.
- `unique()`: shows each distinct value once.

## Simple Baseline Thinking

Before training machine learning models, check simple rules.

Example from the lesson:

```python
df["Many_service_calls"] = (df["Customer service calls"] > 3).astype("int")
```

Then compare this new feature with churn:

```python
pd.crosstab(df["Many_service_calls"], df["Churn"], margins=True)
```

The idea:

- Start with simple summaries and tables.
- Look for obvious patterns.
- Build a simple rule as a baseline.
- More complex models should beat this baseline.
