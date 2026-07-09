# Visual Data Analysis in Python

Source: https://mlcourse.ai/book/topic02/topic02_visual_data_analysis.html

Visual data analysis is a fast way to understand a dataset before building models. It helps reveal distributions, outliers, class imbalance, feature relationships, and possible patterns connected to the target variable.

The article uses a telecom churn dataset. The target is `Churn`, where `True` means the customer left and `False` means the customer stayed.

## Main Feature Types

- **Quantitative features**: ordered numeric values, such as minutes, charges, or number of calls.
- **Categorical features**: values that represent groups, such as `State` or `Area code`.
- **Binary features**: categorical features with exactly two values, such as `Yes`/`No`.
- **Ordinal features**: categorical values with a meaningful order, such as number of customer service calls.

## 1. Univariate Visualization

Univariate analysis studies one feature at a time. The goal is usually to understand the feature's distribution.

### Quantitative Features

**Histogram**

- Splits numeric values into bins.
- Shows the rough distribution shape.
- Useful for spotting skewness, unusual values, or whether a feature looks close to normal.
- Example insight: `Total day minutes` is roughly normal, while `Total intl calls` is right-skewed.

```python
df[["Total day minutes", "Total intl calls"]].hist(figsize=(10, 4))
```

**Density plot / KDE**

- A smoothed version of a histogram.
- Does not depend as strongly on bin size.
- Useful when the overall distribution shape matters more than exact counts.

```python
df[["Total day minutes", "Total intl calls"]].plot(
    kind="density",
    subplots=True,
    layout=(1, 2),
    sharex=False,
    figsize=(10, 4),
)
```

**Box plot**

- Summarizes a numeric distribution with median, quartiles, whiskers, and outliers.
- The box spans from Q1 to Q3.
- `IQR = Q3 - Q1`.
- Whiskers usually cover values within `Q1 - 1.5 * IQR` and `Q3 + 1.5 * IQR`.
- Points outside the whiskers are treated as outliers.

```python
sns.boxplot(x="Total intl calls", data=df)
```

**Violin plot**

- Combines a box-plot idea with a smoothed density shape.
- Useful when distribution shape is important.
- Sometimes it adds little if the box plot already makes the pattern clear.

```python
sns.violinplot(data=df["Total intl calls"])
```

**`describe()`**

- Gives exact numeric statistics: count, mean, standard deviation, min, quartiles, and max.
- Use it together with plots: plots show shape, `describe()` gives exact values.

```python
df[["Total day minutes", "Total intl calls"]].describe()
```

### Categorical and Binary Features

**Frequency table**

- Shows how often each category appears.
- Useful for checking class balance.
- In the churn dataset, the target is imbalanced: many more customers stayed than left.

```python
df["Churn"].value_counts()
```

**Bar plot / count plot**

- Visual version of a frequency table.
- Use for categorical features, not continuous numeric distributions.
- A count plot of `Customer service calls` suggests most customers solve issues within 2 or 3 calls.

```python
sns.countplot(x="Churn", data=df)
sns.countplot(x="Customer service calls", data=df)
```

## 2. Multivariate Visualization

Multivariate analysis studies relationships between two or more features. The correct plot depends on the feature types.

### Quantitative vs. Quantitative

**Correlation matrix**

- Shows pairwise linear relationships between numeric features.
- Helpful for finding redundant or highly related variables.
- Example: call charges are directly calculated from call minutes, so they may not add new information.
- Important because some models, such as linear and logistic regression, can be sensitive to highly correlated inputs.

```python
corr_matrix = df[numerical].corr()
sns.heatmap(corr_matrix)
```

**Scatter plot**

- Plots two numeric variables on x and y axes.
- Useful for seeing correlation, clusters, outliers, and unusual structure.
- If the cloud is axis-aligned and oval-shaped, the variables are likely weakly correlated.

```python
plt.scatter(df["Total day minutes"], df["Total night minutes"])
```

**Joint plot**

- Scatter plot plus marginal distributions.
- Can also show a 2D density estimate.

```python
sns.jointplot(
    x="Total day minutes",
    y="Total night minutes",
    data=df,
    kind="scatter",
)
```

**Scatterplot matrix / pair plot**

- Shows distributions on the diagonal and pairwise scatter plots elsewhere.
- Useful for a small number of numeric features.
- Can become slow and hard to read with many features.

```python
sns.pairplot(df[numerical])
```

### Quantitative vs. Categorical

Use these plots to compare numeric distributions across groups, especially across target classes.

**Scatter plot with color**

- Add a categorical variable with `hue`.
- Useful for seeing whether classes occupy different regions.

```python
sns.lmplot(
    x="Total day minutes",
    y="Total night minutes",
    data=df,
    hue="Churn",
    fit_reg=False,
)
```

**Grouped box plots**

- Compare numeric feature distributions for each category.
- In the churn example, important differences appear for:
  - `Total day minutes`
  - `Customer service calls`
  - `Number vmail messages`

```python
sns.boxplot(x="Churn", y="Total day minutes", data=df)
```

**Cat plot**

- Useful when one numeric variable is compared across two categorical dimensions.
- Example: compare `Total day minutes` by `Churn` and by number of customer service calls.

```python
sns.catplot(
    x="Churn",
    y="Total day minutes",
    col="Customer service calls",
    data=df[df["Customer service calls"] < 8],
    kind="box",
    col_wrap=4,
)
```

### Categorical vs. Categorical

**Count plot with hue**

- Shows how one categorical feature is distributed inside another category.
- Example insights:
  - Churn increases strongly after 4 or more customer service calls.
  - Customers with an international plan have a much higher churn rate.
  - Voice mail plan does not show the same strong effect.

```python
sns.countplot(x="Customer service calls", hue="Churn", data=df)
sns.countplot(x="International plan", hue="Churn", data=df)
```

**Contingency table / cross tabulation**

- A table showing counts for combinations of categorical variables.
- Useful for checking conditional distributions.
- Be careful when categories have few observations; rates can look extreme just because the sample is small.

```python
pd.crosstab(df["State"], df["Churn"]).T
```

## 3. Whole-Dataset Visualization

Simple histograms and pair plots are not enough when there are many features. They become slow, cluttered, and mostly pairwise.

### Dimensionality Reduction

Dimensionality reduction projects high-dimensional data into fewer dimensions, usually 2D or 3D, while trying to preserve important structure.

- It is usually unsupervised because it creates new features without using the target directly.
- **PCA** is a common linear dimensionality reduction method.
- **t-SNE** is a nonlinear method focused on preserving local neighborhoods.

### t-SNE

t-SNE maps high-dimensional points to 2D so that similar observations stay close together and dissimilar observations tend to stay apart.

Basic preprocessing:

1. Remove the target and non-useful text columns.
2. Encode binary values like `Yes`/`No` as `1`/`0`.
3. Scale features so different units do not dominate the result.

```python
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

X = df.drop(["Churn", "State"], axis=1)
X["International plan"] = X["International plan"].map({"Yes": 1, "No": 0})
X["Voice mail plan"] = X["Voice mail plan"].map({"Yes": 1, "No": 0})

X_scaled = StandardScaler().fit_transform(X)
tsne_repr = TSNE(random_state=17).fit_transform(X_scaled)
```

Plotting with the target color can reveal whether churned customers form visible groups.

```python
plt.scatter(
    tsne_repr[:, 0],
    tsne_repr[:, 1],
    c=df["Churn"].map({False: "blue", True: "orange"}),
    alpha=0.5,
)
```

Important warnings:

- t-SNE can be computationally expensive.
- Results can change with the random seed.
- The plot is useful for intuition, but it should not be treated as proof.
- Any visual pattern should be checked with deeper analysis or modeling.

## Practical Workflow

1. Identify feature types and the target variable.
2. Start with univariate plots to understand each feature.
3. Check skewness, outliers, missing-looking values, and class imbalance.
4. Use multivariate plots to compare features with the target.
5. Use correlation heatmaps to detect redundant numeric features.
6. Use grouped box plots or count plots to find features related to churn.
7. Use dimensionality reduction only for high-level intuition.
8. Treat visual discoveries as hypotheses, not final conclusions.

## Key Takeaways

- Visualization is one of the first steps in exploratory data analysis.
- Use histograms, KDEs, box plots, and `describe()` for numeric features.
- Use frequency tables and count plots for categorical features.
- Use scatter plots, heatmaps, pair plots, grouped box plots, and cross tabs for relationships.
- Always choose the plot based on the feature types.
- Watch out for imbalanced classes and small sample sizes.
- t-SNE can reveal clusters, but it is sensitive and should be interpreted carefully.
