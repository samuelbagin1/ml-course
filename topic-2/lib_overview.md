# Seaborn, Matplotlib, and Plotly Overview

Source: https://mlcourse.ai/book/topic02/topic02_additional_seaborn_matplotlib_plotly.html

This lesson compares common Python visualization tools. The examples use a video game sales dataset with columns such as `Platform`, `Genre`, `Global_Sales`, `Critic_Score`, `User_Score`, and `Rating`.

## Dataset Setup

The article starts by importing visualization libraries and loading the data into a pandas `DataFrame`.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["image.cmap"] = "viridis"
```

The dataset is loaded, rows with missing values are removed, and some columns are converted to correct numeric types.

```python
df = pd.read_csv(DATA_URL + "video_games_sales.csv").dropna()

df["User_Score"] = df["User_Score"].astype("float64")
df["Year_of_Release"] = df["Year_of_Release"].astype("int64")
df["User_Count"] = df["User_Count"].astype("int64")
df["Critic_Count"] = df["Critic_Count"].astype("int64")
```

Key idea: always check `df.info()` after loading data. Some numeric columns may be read as `object`, so they must be converted before plotting or analysis.

## 1. `DataFrame.plot()`

`DataFrame.plot()` is the quickest way to create simple plots directly from pandas.

- It is built on top of Matplotlib.
- It is useful for quick EDA without much setup.
- It works well after grouping or aggregating data.
- The `kind` argument changes the chart type.

Example: total sales by year.

```python
df[[x for x in df.columns if "Sales" in x] + ["Year_of_Release"]] \
    .groupby("Year_of_Release") \
    .sum() \
    .plot()
```

Example: same data as a bar chart.

```python
df[[x for x in df.columns if "Sales" in x] + ["Year_of_Release"]] \
    .groupby("Year_of_Release") \
    .sum() \
    .plot(kind="bar", rot=45)
```

Use `DataFrame.plot()` when you need a fast chart from a DataFrame and do not need advanced styling.

## 2. Seaborn

Seaborn is a higher-level visualization library built on Matplotlib.

- It has better default styles.
- It needs less code for statistical plots.
- It is especially useful for distributions, relationships, grouped comparisons, and heatmaps.
- `sns.set()` improves the default appearance of plots.

### `pairplot()`

`pairplot()` shows pairwise relationships between several numeric variables.

- Diagonal cells show feature distributions.
- Other cells show scatter plots between pairs of features.
- Useful for quickly checking correlations, clusters, and unusual patterns.
- Can become slow with many rows or many columns.

```python
sns.pairplot(
    df[["Global_Sales", "Critic_Score", "Critic_Count", "User_Score", "User_Count"]]
)
```

### `histplot()`

`histplot()` shows the distribution of one numeric feature.

- Use it to see shape, skewness, and spread.
- Add `kde=True` to show a smoothed density curve.
- `stat="density"` normalizes bars to density instead of raw counts.

```python
sns.histplot(df["Critic_Score"], kde=True, stat="density")
```

### `jointplot()`

`jointplot()` combines a scatter plot with marginal distributions.

- It is useful for studying two numeric variables together.
- The center shows the relationship.
- The sides show each variable's distribution.

```python
sns.jointplot(
    x="Critic_Score",
    y="User_Score",
    data=df,
    kind="scatter",
)
```

### `boxplot()`

`boxplot()` compares numeric distributions across categories.

- The box shows the interquartile range from Q1 to Q3.
- The line inside the box is the median.
- Whiskers usually cover values within `1.5 * IQR`.
- Points outside the whiskers are outliers.

Example: critic scores across top gaming platforms.

```python
top_platforms = df["Platform"].value_counts().head(5).index

sns.boxplot(
    y="Platform",
    x="Critic_Score",
    data=df[df["Platform"].isin(top_platforms)],
    orient="h",
)
```

### `heatmap()`

`heatmap()` displays numeric values using color intensity.

- Useful for tables, matrices, and pivot tables.
- Good for seeing large values and patterns quickly.
- In the article, it shows global sales by platform and genre.

```python
platform_genre_sales = (
    df.pivot_table(
        index="Platform",
        columns="Genre",
        values="Global_Sales",
        aggfunc="sum",
    )
    .fillna(0)
)

sns.heatmap(platform_genre_sales, annot=True, fmt=".1f", linewidths=0.5)
```

## 3. Plotly

Plotly is used for interactive plots.

- It works well in Jupyter notebooks.
- Users can hover to see exact values.
- Users can zoom, pan, and hide/show chart series.
- Plots can be saved as standalone HTML files.

The main Plotly object is `Figure`. A figure contains:

- `data`: one or more traces, such as lines, bars, or boxes.
- `layout`: chart title, axes, labels, and style settings.

### Line Plot

Line plots are useful for trends over time.

Example: game sales and number of releases by year.

```python
import plotly.graph_objs as go

trace0 = go.Scatter(
    x=years_df.index,
    y=years_df["Global_Sales"],
    name="Global Sales",
)

trace1 = go.Scatter(
    x=years_df.index,
    y=years_df["Number_of_Games"],
    name="Number of games released",
)

fig = go.Figure(
    data=[trace0, trace1],
    layout={"title": "Statistics for video games"},
)
```

### Bar Chart

Bar charts compare categories.

Example: compare platforms by global sales and number of games.

```python
trace0 = go.Bar(
    x=platforms_df.index,
    y=platforms_df["Global_Sales"],
    name="Global Sales",
)

trace1 = go.Bar(
    x=platforms_df.index,
    y=platforms_df["Number_of_Games"],
    name="Number of games released",
)
```

### Box Plot

Plotly can also create interactive box plots.

Example: critic score distribution by game genre.

```python
data = []

for genre in df.Genre.unique():
    data.append(go.Box(y=df[df.Genre == genre].Critic_Score, name=genre))
```

## Which Library to Use?

- Use **pandas `DataFrame.plot()`** for quick plots from grouped DataFrames.
- Use **Matplotlib** when you need low-level control over every chart detail.
- Use **Seaborn** for clean statistical plots with less code.
- Use **Plotly** when interaction matters, such as hover values, zooming, or hiding series.

## Key Takeaways

- Pandas plotting is fast and convenient for simple EDA.
- Seaborn gives strong default styling and high-level statistical plots.
- Matplotlib is the base layer behind pandas plots and Seaborn.
- Plotly is best when the viewer should interact with the chart.
- Always check data types before plotting, especially after loading CSV files.
- Choose the plot based on the question: trend, distribution, relationship, category comparison, or matrix pattern.
