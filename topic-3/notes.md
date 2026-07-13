# Classification, Decision Trees & k-Nearest Neighbors

Source: [mlcourse.ai — Topic 3](https://mlcourse.ai/book/topic03/topic03_decision_trees_kNN.html)

## 1. Supervised learning: classification and regression

In supervised learning, each training example has:

- **features** $X$: the input measurements used for prediction, e.g. age, income, or account usage;
- **target** $y$: the known answer the model learns to predict.

The goal is to learn a rule from the training data that also works on unseen data.

- **Classification** predicts a discrete class. Example: predict whether a customer will default on a loan (`default` / `no default`). Binary classification has two possible classes; multiclass classification has more.
- **Regression** predicts a continuous number. Example: predict the number of days a payment will be overdue.

For classification, a simple evaluation metric is **accuracy**:

$$
\text{accuracy} = \frac{\text{number of correct predictions}}{\text{total number of predictions}}
$$

Accuracy is useful when classes are reasonably balanced, but can be misleading for strongly imbalanced data. For example, a model that always predicts “no churn” can seem accurate when churn is rare, while failing to identify any churners.

---

## 2. Decision trees

A **decision tree** learns a sequence of if/else questions. Each internal node tests one feature, each branch represents an answer to the test, and each leaf makes a prediction.

Example rule:

```text
if income <= 5000 and home_ownership == "no":
    predict loan denial
else:
    predict loan approval
```

### Why trees are useful

Trees are easy to explain: to understand one prediction, follow the path from the root to a leaf. This is especially valuable when a person needs to inspect or justify a model decision.

For a classification tree, each leaf usually predicts the **majority class** among the training examples that reached it. A tree divides the feature space into rectangular regions, one region per leaf.

### Choosing a split

At a node, the tree considers possible questions such as:

```text
age <= 30?
income <= 5000?
owns_home == yes?
```

It selects the split that makes the resulting child groups as pure as possible. A pure group contains examples from only one class.

#### Entropy

**Entropy** measures uncertainty (or class mixing) in a node:

$$
H(S) = -\sum_{k=1}^{K} p_k \log_2(p_k)
$$

where $p_k$ is the fraction of class $k$ in the node.

- Entropy is $0$ for a pure node because there is no uncertainty.
- In binary classification, entropy is largest when both classes are equally frequent ($p = 0.5$).

#### Information gain

The quality of a split is the reduction in uncertainty it produces:

$$
IG = H(\text{parent}) - \sum_{j=1}^{m}\frac{n_j}{n}H(\text{child}_j)
$$

The tree prefers the split with the largest **information gain**. The child entropies are weighted by the proportion of examples that enter each child, so a tiny pure group cannot make a poor overall split look excellent.

#### Other impurity criteria

- **Gini impurity**:

  $$
  G = 1 - \sum_{k=1}^{K}p_k^2
  $$

  This is the usual default in scikit-learn. It behaves similarly to entropy for many classification problems.

- **Misclassification error**:

  $$
  E = 1 - \max_k p_k
  $$

  It is less sensitive to changes in class proportions, so it is rarely used to grow trees.

### How trees handle numerical features

For a numerical feature, a tree learns a threshold such as `age <= 30.0`. A useful candidate threshold lies between consecutive sorted values where the target class changes. The algorithm scores the candidates and retains the best one.

This means trees do not need feature scaling: changing income from euros to cents changes the numerical threshold but not the logical split. This is very different from distance-based methods such as k-NN.

### Greedy tree-building algorithm

Decision-tree algorithms are greedy: they make the best local split at the current node, then repeat recursively for each child.

```text
build(node_data):
    if stopping criterion is met:
        create a leaf prediction
    else:
        find the best split of node_data
        build the left subtree
        build the right subtree
```

The globally optimal tree is difficult to find, so greedy splitting does not guarantee the best possible entire tree.

### Overfitting and regularization

A fully grown tree can memorize random details of the training set. It may achieve perfect training accuracy but make poor predictions on new data: this is **overfitting**.

Common ways to limit complexity:

- `max_depth`: stop the tree after a fixed number of levels;
- `min_samples_leaf`: require a minimum number of samples in every leaf;
- `min_samples_split`: require enough samples before splitting a node;
- `max_features`: consider only a limited number of features at each split;
- **pruning**: grow a large tree, then remove branches that do not improve validation performance.

These hyperparameters should be chosen with cross-validation, not by maximizing training accuracy.

### Decision-tree regression

A regression tree uses the same splitting idea, but searches for splits that reduce target variance (equivalently, commonly minimize mean squared error). A leaf predicts the mean target value of its training examples.

Therefore, a regression tree produces a **piecewise-constant** prediction: it is flat inside each leaf region. It can interpolate patterns within the training range but cannot reliably extrapolate beyond it.

### scikit-learn

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(
    criterion="gini",      # or "entropy" / "log_loss"
    max_depth=5,
    min_samples_leaf=10,
    random_state=42,
)
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)
```

---

## 3. k-Nearest Neighbors (k-NN)

**k-Nearest Neighbors** predicts from the labels of the most similar training examples. Its central assumption is the **compactness hypothesis**: examples close to one another in a suitable feature space tend to have similar targets.

To classify a new example:

1. Calculate its distance to every training example.
2. Select the $k$ closest examples.
3. Predict the most common class among those neighbors.

For regression, predict the mean or median target value of the $k$ neighbors instead of a class vote.

### A lazy learner

k-NN does not fit a compact model during training; it mostly stores the data. The expensive work happens at prediction time, when it has to search for neighbors. This contrasts with a decision tree, which spends effort learning the tree and then predicts quickly by following a short path.

### Important hyperparameters

- `n_neighbors` ($k$):
  - small $k$ gives a very local, flexible boundary but is sensitive to noise and outliers;
  - large $k$ smooths the boundary but can miss local structure.
- `metric`: the definition of similarity, e.g. Euclidean, Manhattan, cosine, or a domain-specific metric.
- `weights`:
  - `uniform`: every neighbor has one equal vote;
  - `distance`: closer neighbors have more influence.

### Feature scaling is essential

k-NN is based on distance. If income ranges from 0–100,000 and age ranges from 0–100, income will dominate Euclidean distance purely because it uses larger numbers. Standardize numerical features before k-NN:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

knn = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance")),
])
knn.fit(X_train, y_train)
```

Putting scaling inside a `Pipeline` prevents data leakage: the scaler is fitted only on the training fold during cross-validation.

### Curse of dimensionality

As the number of features grows, distances between points become less informative: even the “nearest” point may not be very similar. Irrelevant/noisy features can drown out a genuinely useful feature. This is the **curse of dimensionality**, and it is a major limitation of k-NN.

Feature selection, sensible feature weighting, dimensionality reduction, or a better distance metric can help.

---

## 4. Evaluating models and tuning hyperparameters

The real objective is **generalization**: good performance on data the model did not train on.

### Hold-out set

Split the data into:

- a **training set** for fitting models;
- a **hold-out test set** that is not used to choose models or hyperparameters.

The hold-out set is saved for the final, unbiased estimate of model performance.

### k-fold cross-validation

With **$k$-fold cross-validation**, divide the training data into $k$ folds. Train $k$ times, each time using $k - 1$ folds for training and the remaining fold for validation. Average the $k$ validation scores.

Cross-validation is more reliable than relying on a single split, though it costs more computation. Use stratified folds for classification so each fold keeps approximately the same class balance.

### Grid search

`GridSearchCV` evaluates several hyperparameter combinations with cross-validation and keeps the best one.

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

tree_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid={
        "max_depth": [3, 5, 8, None],
        "min_samples_leaf": [1, 5, 10, 20],
        "max_features": [None, "sqrt", "log2"],
    },
    cv=cv,
    scoring="accuracy",
)
tree_search.fit(X_train, y_train)

print(tree_search.best_params_)
print(tree_search.best_score_)
```

For k-NN, tune the number of neighbors, distance metric, and voting weights. Scaling must remain inside the pipeline so that each cross-validation fold learns its scaling values only from that fold's training data.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

knn_search = GridSearchCV(
    estimator=Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ]),
    param_grid={
        "knn__n_neighbors": range(1, 32, 2),
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    },
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
)
knn_search.fit(X_train, y_train)

print(knn_search.best_params_)
print(knn_search.best_score_)
```

Use odd values of $k$ in binary classification to reduce the chance of tied votes. The best value of $k$ is data-dependent, so select it from the cross-validation score rather than assuming a standard choice.

Workflow:

1. Split off a final test/hold-out set.
2. Use cross-validation on the training portion to choose hyperparameters.
3. Refit the chosen model on all training data.
4. Evaluate the final model once on the untouched hold-out set.

---

## 5. Decision trees vs. k-NN

| Aspect | Decision tree | k-NN |
| --- | --- | --- |
| Main idea | Learn a sequence of feature-based rules | Vote/average among similar training examples |
| Training time | Usually fast | Very small (mostly stores data) |
| Prediction time | Usually fast | Can be expensive on large datasets |
| Scaling needed | No | Yes, for most distance metrics |
| Interpretability | High; inspect rules and paths | Good for small $k$, weaker for many neighbors |
| Numerical/categorical data | Supports both (categoricals need encoding in sklearn) | Requires a meaningful numeric representation and distance |
| Main risk | Overfitting and instability | Wrong distance metric, outliers, high dimensionality |
| Decision boundary | Axis-aligned rectangular regions | Flexible, locally shaped regions |

### Decision-tree strengths

- Clear, human-readable rules and visualizations.
- Fast fitting and prediction.
- Handles different feature scales without preprocessing.
- Can model non-linear interactions.

### Decision-tree limitations

- A small change to the data can produce a very different tree.
- Greedy splitting can miss a globally better tree.
- Axis-aligned splits can approximate a diagonal boundary inefficiently.
- Trees easily overfit unless constrained or pruned.
- Single trees do not extrapolate in regression.

### k-NN strengths

- Simple and useful as a baseline.
- Naturally handles complex local decision boundaries.
- Can be used for classification, regression, and recommendation-like tasks.
- Flexible when paired with a well-designed similarity measure.

### k-NN limitations

- Prediction can be slow and memory-intensive for large training sets.
- Sensitive to scale, feature selection, distance metric, and $k$.
- Small $k$ is noise-sensitive; large $k$ can underfit.
- Often degrades in high-dimensional spaces.

## 6. Practical checklist

1. Begin with a clear target and an appropriate metric.
2. Keep a final hold-out set untouched until the end.
3. Try both a constrained decision tree and a scaled k-NN pipeline as interpretable baselines.
4. Tune `max_depth` / leaf size for trees and $k$ / metric / weights for k-NN using cross-validation.
5. Compare cross-validation performance with hold-out performance. A large gap suggests overfitting or leakage.
6. Inspect errors and use domain knowledge: a better feature representation or distance metric can matter more than changing algorithms.
