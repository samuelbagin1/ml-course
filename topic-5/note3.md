# Random Forest: Tuning and Best Parameters

This note focuses on the practical application, tuning, and mathematical intuition of Random Forest parameters, specifically addressing variance reduction and bias.

## 1. Variance and Decorrelation

The main strength of a Random Forest lies in its ability to dramatically reduce the variance of the predictions compared to a single Decision Tree. 

The variance of a Random Forest ensemble $f(x)$ is defined by the following formula:

$$ \text{Var}(f(x)) = \rho(x)\sigma^2(x) + \frac{1 - \rho(x)}{B}\sigma^2(x) $$

where:
- **$\text{Var}(f(x))$**: The variance of the final Random Forest prediction.
- **$B$**: The number of trees in the forest (controlled by `n_estimators`).
- **$\sigma^2(x)$**: The sample variance of any randomly selected single tree.
- **$\rho(x)$**: The sample correlation coefficient between any two trees used in the averaging process, estimated on the input $x$.

### Intuition:
As the number of trees $B$ increases, the second term $\frac{1 - \rho(x)}{B}\sigma^2(x)$ shrinks toward zero. 
However, the first term $\rho(x)\sigma^2(x)$ remains. This shows that the variance of the ensemble is heavily dependent on the **correlation $\rho(x)$** between the trees.

To minimize this correlation, Random Forest uses the **Random Subspace Method** (controlled by `max_features`): at each split, it only considers a random subset of features. This forces the trees to look different from each other, lowering $\rho(x)$ and, consequently, the total model variance.

## 2. Bias

The bias of a Random Forest is inherently tied to the bias of its individual trees.

$$ \text{Bias} = \mathbb{E}[T(x,\Theta(Z))] - y(x) $$

where:
- **$\mathbb{E}[T(x,\Theta(Z))]$**: The expected prediction of a single tree $T$ trained on a random bootstrap sample $\Theta(Z)$.
- **$y(x)$**: The true target value.

The bias of a Random Forest is roughly the same (or slightly higher due to feature restriction) as the bias of a single, unpruned decision tree. Therefore, **the improvement in prediction accuracy obtained by Random Forests is solely the result of variance reduction**, not bias reduction.

---

## 3. Key Hyperparameters to Tune

When building a `RandomForestClassifier` or `RandomForestRegressor` in Scikit-Learn, focus on the following parameters:

- `n_estimators`: The number of trees in the forest. More trees reduce variance but increase computational cost.
- `max_depth`: The maximum depth of the tree. Acts as a strong regularizer.
- `min_samples_leaf`: The minimum number of samples required to be at a leaf node. Prevents the tree from building nodes that fit to a single noise point.
- `max_features`: The number of features to consider when looking for the best split (usually $\sqrt{d}$ for classification, where $d$ is total features).
- `criterion`: The function to measure the quality of a split (`gini` or `entropy`).

---

## 4. Practice: Tuning Random Forests in a Real Problem

Here is a complete example of predicting customer churn (classification), showing how to validate performance and systematically search for the best hyperparameters using Validation Curves and Grid Search.

### Step 1: Loading Data and Baseline Evaluation

```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load data
DATA_PATH = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"
df = pd.read_csv(DATA_PATH + "telecom_churn.csv")

# Choose only numeric features for simplicity
cols = []
for i in df.columns:
    if (df[i].dtype == "float64") or (df[i].dtype == 'int64'):
        cols.append(i)

# Divide the dataset into the input and target
X, y = df[cols].copy(), np.asarray(df["Churn"], dtype='int8')

# Initialize a stratified split of our dataset for the validation process
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the classifier with the default parameters
rfc = RandomForestClassifier(random_state=42, n_jobs=-1)

# Train and evaluate the baseline model
results = cross_val_score(rfc, X, y, cv=skf)
print("CV accuracy score: {:.2f}%".format(results.mean() * 100))
# CV accuracy score: 92.35%
```

### Step 2: Exploring Individual Parameters

We iterate over different values of parameters to see their effect on Cross-Validation accuracy.

**Tuning `n_estimators` (Number of trees):**
```python
train_acc = []
test_acc = []
trees_grid = [5, 10, 15, 20, 30, 50, 75, 100]

for ntrees in trees_grid:
    rfc = RandomForestClassifier(n_estimators=ntrees, random_state=42, n_jobs=-1)
    
    temp_train_acc, temp_test_acc = [], []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        rfc.fit(X_train, y_train)
        temp_train_acc.append(rfc.score(X_train, y_train))
        temp_test_acc.append(rfc.score(X_test, y_test))
        
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)

train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
print("Best CV accuracy is {:.2f}% with {} trees".format(
    max(test_acc.mean(axis=1)) * 100, 
    trees_grid[np.argmax(test_acc.mean(axis=1))]
))
# Best CV accuracy is 92.50% with 75 trees
```

**Tuning `max_depth` (Maximum depth of the trees):**
Fixing `n_estimators` to 100 and testing depths:
```python
train_acc = []
test_acc = []
max_depth_grid = [3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24]

for max_depth in max_depth_grid:
    rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=max_depth)
    
    temp_train_acc, temp_test_acc = [], []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        rfc.fit(X_train, y_train)
        temp_train_acc.append(rfc.score(X_train, y_train))
        temp_test_acc.append(rfc.score(X_test, y_test))
        
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)

train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
print("Best CV accuracy is {:.2f}% with {} max_depth".format(
    max(test_acc.mean(axis=1)) * 100, 
    max_depth_grid[np.argmax(test_acc.mean(axis=1))]
))
# Best CV accuracy is 92.50% with 22 max_depth
```

**Tuning `min_samples_leaf`:**
```python
train_acc = []
test_acc = []
min_samples_leaf_grid = [1, 3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24]

for min_samples_leaf in min_samples_leaf_grid:
    rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=min_samples_leaf)
    
    temp_train_acc, temp_test_acc = [], []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        rfc.fit(X_train, y_train)
        temp_train_acc.append(rfc.score(X_train, y_train))
        temp_test_acc.append(rfc.score(X_test, y_test))
        
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)

train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
print("Best CV accuracy is {:.2f}% with {} min_samples_leaf".format(
    max(test_acc.mean(axis=1)) * 100, 
    min_samples_leaf_grid[np.argmax(test_acc.mean(axis=1))]
))
# Best CV accuracy is 92.35% with 1 min_samples_leaf
```

**Tuning `max_features`:**
```python
train_acc = []
test_acc = []
max_features_grid = [2, 4, 6, 8, 10, 12, 14, 16]

for max_features in max_features_grid:
    rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_features=max_features)
    
    temp_train_acc, temp_test_acc = [], []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        rfc.fit(X_train, y_train)
        temp_train_acc.append(rfc.score(X_train, y_train))
        temp_test_acc.append(rfc.score(X_test, y_test))
        
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)

train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
print("Best CV accuracy is {:.2f}% with {} max_features".format(
    max(test_acc.mean(axis=1)) * 100, 
    max_features_grid[np.argmax(test_acc.mean(axis=1))]
))
# Best CV accuracy is 92.41% with 14 max_features
```

### Step 3: Finding the Optimal Combination using GridSearchCV

After exploring how individual parameters affect learning curves, the best approach is to use `GridSearchCV` to exhaustively test combinations of these parameters.

```python
# Initialize the set of parameters for exhaustive search
parameters = {
    'max_features': [4, 7, 10, 13], 
    'min_samples_leaf': [1, 3, 5, 7], 
    'max_depth': [5, 10, 15, 20]
}

# Instantiate the model with fixed n_estimators
rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Setup GridSearch
gcv = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)

# Fit over all combinations
gcv.fit(X, y)

# Output the best parameters and corresponding score
print(gcv.best_params_, gcv.best_score_)
# Output:
# {'max_depth': 20, 'max_features': 10, 'min_samples_leaf': 3} 0.9252968110539325
```
By performing a grid search, we found that combining `max_depth=20`, `max_features=10`, and `min_samples_leaf=3` yielded the absolute best cross-validation accuracy of ~92.53%.


## 5. Pros and Cons of Random Forests

**Pros:**
- **High prediction accuracy**: Often performs better than linear algorithms and is comparable to boosting.
- **Robust to outliers**: Thanks to random sampling.
- **Insensitive to scaling**: Feature scaling and monotonic transformations do not affect the random subspace selection or tree splits.
- **Works well out-of-the-box**: Doesn't require extremely fine-grained parameter tuning to get decent results (though tuning can add 0.5–3% accuracy).
- **Efficient on large datasets**: Handles many features and classes well.
- **Handles mixed data**: Can process both continuous and discrete variables equally well.
- **Rarely overfits**: Adding more trees almost always improves the composition until it reaches an asymptote.
- **Feature importance**: Built-in methods to estimate the significance of variables.
- **Handles missing data well**: Maintains good accuracy even when a large part of the data is missing.
- **Parallelizable**: Easily scales across multiple cores.
- **Versatile**: Can be extended to unsupervised clustering, data visualization, and outlier detection.

**Cons:**
- **Difficult to interpret**: The output is a "black box" compared to a single decision tree.
- **No formal p-values**: Lack of formal statistical tests for feature significance estimation.
- **Poor on sparse data**: Performs worse than linear methods on very sparse data (e.g., bag of words, text inputs).
- **Cannot extrapolate**: Unable to predict values outside the range of the training data (though this also means outliers don't cause extreme predictions).
- **Prone to overfitting on noisy data**: Can overfit in specific problems with highly noisy datasets.
- **Categorical variable bias**: Favors categorical variables with a greater number of levels because they offer more splitting possibilities to gain accuracy.
- **Correlation bias**: If a dataset contains groups of correlated features, it might preferentially choose groups of smaller size.
- **Resource heavy**: The resulting model is large and requires a significant amount of RAM.

---

## 6. Transformation of a Dataset into a High-Dimensional Representation

While Random Forests are primarily used for supervised learning, they can also be applied in an unsupervised setting using Scikit-Learn’s `RandomTreesEmbedding`.

This technique transforms a dataset into a high-dimensional, sparse representation:
1. **Build Trees**: It first builds Extremely Randomized Trees entirely unsupervised.
2. **Binary Feature Extraction**: It then uses the index of the leaf containing the example as a new feature.
   - For example, if an input instance falls into the first leaf of a tree, it is assigned a feature value of `1`. If not, it gets a `0`.
   - This creates a massive, sparse binary matrix.

**Why is this useful?**
Because nearby data points are highly likely to fall into the same leaf node across different trees, this transformation provides an implicit, nonparametric estimate of their density. You can control the number of features and the sparseness of the data by simply adjusting the number of trees and their maximum depth.
