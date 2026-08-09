# Linear Regression and Classification

Sources: [Part 1: OLS](https://mlcourse.ai/book/topic04/topic4_linear_models_part1_mse_likelihood_bias_variance.html), [Part 2: logistic regression](https://mlcourse.ai/book/topic04/topic4_linear_models_part2_logit_likelihood_learning.html), [Part 3: regularization](https://mlcourse.ai/book/topic04/topic4_linear_models_part3_regul_example.html), [Part 4: examples](https://mlcourse.ai/book/topic04/topic4_linear_models_part4_good_bad_logit_movie_reviews_XOR.html), and [Part 5: curves](https://mlcourse.ai/book/topic04/topic4_linear_models_part5_valid_learning_curves.html).

## 1. The main idea of linear models

A **linear model** combines input features using learned weights. With an intercept (bias) term, its prediction is

$$
\hat y = w_0 + w_1x_1 + \dots + w_mx_m = \mathbf{w}^T\mathbf{x}.
$$

The same linear score is used differently depending on the task:

- **linear regression** predicts a continuous value, such as a house price;
- **linear classification** converts the score into a class or class probability, such as churn / no churn.

Example: a price model may learn $\hat y = 50{,}000 + 2{,}000 \cdot \text{area}$, so a 60 m$^2$ flat is predicted to cost $170{,}000$. The coefficient means that one additional m$^2$ changes the prediction by $2{,}000$, holding other features fixed.

---

## 2. Linear regression and Ordinary Least Squares (OLS)

For $n$ observations, write the model as

$$
\mathbf{y} = \mathbf{Xw} + \boldsymbol{\epsilon},
$$

where $\mathbf{X}$ is the feature matrix, $\mathbf{w}$ contains the weights, and $\boldsymbol{\epsilon}$ is unexplained noise.

**OLS** chooses weights that minimize the average squared residual:

$$
\operatorname{MSE}(\mathbf{w}) =
\frac{1}{n}\sum_{i=1}^{n}(y_i - \mathbf{w}^T\mathbf{x}_i)^2
= \frac{1}{n}\lVert\mathbf{y} - \mathbf{Xw}\rVert_2^2.
$$

Squaring makes large errors much more costly than small errors and gives a smooth objective that is easy to optimize. If $\mathbf{X}^T\mathbf{X}$ is invertible, the closed-form OLS solution is

$$
\hat{\mathbf{w}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}.
$$

In practice, libraries use numerically stable solvers; for very large datasets, gradient-based optimization is often preferable to directly inverting a matrix.

### Why MSE appears: maximum likelihood

Assume independent errors are Gaussian:

$$
\epsilon_i \sim \mathcal{N}(0, \sigma^2).
$$

Then $y_i \mid \mathbf{x}_i$ is normally distributed around $\mathbf{w}^T\mathbf{x}_i$. Its log-likelihood is, up to constants,

$$
\log p(\mathbf{y}\mid\mathbf{X},\mathbf{w})
= -\frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \mathbf{w}^T\mathbf{x}_i)^2 + \text{constant}.
$$

Therefore, **maximizing likelihood is equivalent to minimizing squared error** under the Gaussian-noise assumption.

---

## 3. Bias, variance, and regularization

For a true process $y = f(\mathbf{x}) + \epsilon$, expected prediction error decomposes into

$$
\mathbb{E}\left[(y - \hat f(\mathbf{x}))^2\right]
= \operatorname{Bias}(\hat f)^2
+ \operatorname{Var}(\hat f)
+ \sigma^2.
$$

- **Bias**: systematic error from a model that is too simple. A straight line fitted to a curved relationship has high bias (**underfitting**).
- **Variance**: sensitivity to which training examples were sampled. A very flexible model that changes greatly after a few data changes has high variance (**overfitting**).
- $\sigma^2$ is irreducible noise; no model can remove it.

The goal is not the smallest training error; it is the best bias--variance trade-off on new data.

### Ridge ($L_2$) regularization

Highly correlated features can make OLS weights unstable. Ridge regression adds a penalty for large weights:

$$
J(\mathbf{w}) =
\frac{1}{n}\lVert\mathbf{y}-\mathbf{Xw}\rVert_2^2
+ \lambda\lVert\mathbf{w}\rVert_2^2.
$$

It deliberately adds some bias to reduce variance. Larger $\lambda$ shrinks weights more strongly; normally the intercept is not penalized. In scikit-learn notation, logistic regression uses $C = 1/\lambda$: a **smaller** $C$ means stronger regularization.

---

## 4. Linear classification and logistic regression

A binary linear classifier separates classes with the hyperplane

$$
\mathbf{w}^T\mathbf{x} = 0.
$$

It can predict a label with $\operatorname{sign}(\mathbf{w}^T\mathbf{x})$, but a raw linear score is not a probability. **Logistic regression** applies the sigmoid function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad
P(y=1\mid\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x}).
$$

Equivalently, it models the **log-odds** as linear:

$$
\log\frac{p}{1-p} = \mathbf{w}^T\mathbf{x}.
$$

Example: if a churn model outputs $p=0.80$, a threshold of $0.5$ predicts churn. The threshold can be changed when false positives and false negatives have different business costs; the learned probabilities do not need to change.

### Logistic loss and margin

With $y_i \in \{0,1\}$ and $p_i = \sigma(\mathbf{w}^T\mathbf{x}_i)$, negative log-likelihood (binary cross-entropy) is

$$
\mathcal{L}(\mathbf{w}) = -\sum_{i=1}^{n}
\left[y_i\log p_i + (1-y_i)\log(1-p_i)\right].
$$

This penalizes confident wrong predictions heavily. Using labels $y_i\in\{-1,+1\}$, the equivalent loss is

$$
\mathcal{L}_{\log} =
\sum_{i=1}^{n}\log\left(1+e^{-y_i\mathbf{w}^T\mathbf{x}_i}\right).
$$

Here $M_i = y_i\mathbf{w}^T\mathbf{x}_i$ is the **margin**: positive means correct classification, while a larger positive margin means a more confident prediction.

### Regularized logistic regression

$$
J(\mathbf{w}) = \mathcal{L}_{\log}(\mathbf{w}) + \lambda\lVert\mathbf{w}\rVert_2^2
= \mathcal{L}_{\log}(\mathbf{w}) + \frac{1}{C}\lVert\mathbf{w}\rVert_2^2.
$$

- small $C$ / large $\lambda$: very smooth, simple boundary; too strong can underfit;
- large $C$ / small $\lambda$: flexible boundary and large weights; too weak can overfit.

Choose $C$ with cross-validation, not training accuracy.

---

## 5. Nonlinear patterns with polynomial features

The classifier is always linear in its **input features**, but those features may be transformations. For two original features, degree-2 polynomial features are

$$
1,\ x_1,\ x_2,\ x_1^2,\ x_1x_2,\ x_2^2.
$$

Logistic regression can learn a plane in this expanded space; when viewed in the original $(x_1,x_2)$ space, its boundary can be curved.

This solves patterns such as **XOR**, where diagonal classes cannot be separated by one straight line. However, the number of polynomial terms grows quickly, so high degrees are expensive and prone to overfitting. Regularization is especially important after feature expansion.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("logit", LogisticRegression(C=1.0, max_iter=1000)),
])
model.fit(X_train, y_train)
```

The pipeline prevents leakage: scaling and feature construction are fitted separately inside each training fold during cross-validation.

---

## 6. Why logistic regression works well for text

With **bag of words**, each vocabulary word becomes a feature. A review such as “great acting, boring plot” becomes a sparse vector of word counts or indicators. Logistic regression adds the learned contribution of every present word:

$$
\operatorname{logit}(p) = w_0 + \sum_j w_jx_j.
$$

For sentiment analysis, positive weights may be learned for words such as `excellent`, and negative weights for `boring`. This makes logistic regression fast, interpretable, and effective for high-dimensional sparse text. It cannot directly understand word order or negation; `not good` is difficult with word counts alone.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

text_model = make_pipeline(
    CountVectorizer(),
    LogisticRegression(C=0.1, max_iter=1000),
)
text_model.fit(text_train, y_train)
```

Put `CountVectorizer` inside the pipeline so the vocabulary is learned from training data only.

---

## 7. Validation and learning curves

A **validation curve** plots training and cross-validation performance while one hyperparameter changes (for example, $C$, $\lambda$, polynomial degree, or tree depth).

- Training and validation scores both poor and close: **underfitting**. Use a more expressive model, better features, or weaker regularization.
- Training score much better than validation: **overfitting**. Use stronger regularization, a simpler model, or more data.
- Best validation score: a good hyperparameter region; confirm once on a final untouched test set.

A **learning curve** plots performance against the number of training examples while keeping the model fixed.

- Curves converge at a poor score: more data alone is unlikely to help; reduce bias by changing features or model complexity.
- Validation score is still improving and the curves have not converged: collecting more data may help reduce variance.

Training performance alone is not evidence of a useful model. Cross-validation estimates how well it should generalize to unseen data.
