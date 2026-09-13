# Topic 6: Feature Engineering and Feature Selection

Source: [mlcourse.ai - Topic 6](https://mlcourse.ai/book/topic06/topic6_feature_engineering_feature_selection.html)

These notes follow every section and headline in the article. The central message is **garbage in, garbage out**: a simple model trained on carefully prepared data can beat a complex ensemble trained on poor data.

## Article outline (introductory outline)

The article distinguishes three related but different jobs:

1. **Feature extraction / feature engineering** - convert raw data into features that a model can consume.
2. **Feature transformation** - change the representation or distribution of existing features to make learning more effective.
3. **Feature selection** - remove features that are redundant, noisy, computationally expensive, or harmful to generalization.

The examples use the Renthop / Two Sigma Connect rental-listing dataset. The target is a three-class popularity label, `['low', 'medium', 'high']`, evaluated with **log loss** (lower is better).

Typical data-loading setup:

```python
import pandas as pd
from pathlib import Path

DATA_PATH = Path("../../_static/data/renthop_train.json.gz")
df = pd.read_json(DATA_PATH, compression="gzip", convert_dates=["created"])
```

## Article outline (detailed outline, `id1`)

### Feature Extraction

1. Texts
2. Images
3. Geospatial data
4. Date and time
5. Time series, web, etc.

### Feature transformations

1. Normalization and changing distribution
2. Interactions
3. Filling in the missing values

### Feature selection

1. Statistical approaches
2. Selection by modeling
3. Grid search

---

## Feature Extraction

Real data rarely arrives as a ready-to-use numeric matrix. Feature extraction is therefore the first practical step: identify the information in raw text, pixels, coordinates, timestamps, logs, or metadata and encode it as model features. Reading a clean CSV directly into a `numpy.array` is the unusual easy case.

### Texts

#### 1. Tokenization

Tokenization splits text into meaningful units (tokens). A token is often a word, but naive splitting can lose meaning: **"Santa Barbara"** is a phrase that may be one token, while **"rock'n'roll"** should not be broken into unrelated pieces. Language-aware tokenizers help, but domain-specific sources (slang, misspellings, newspapers, typos) still require inspection.

#### 2. Text normalization

After tokenization, normalize word forms with:

- **Stemming** - heuristically removes word endings.
- **Lemmatization** - maps words to a dictionary/base form using linguistic information.

Both reduce the number of distinct terms, but they make different accuracy/speed trade-offs.

#### 3. Bag of Words (BoW)

Build a vocabulary and represent each document by a vector whose length equals the vocabulary size. The value at position `j` is the count of vocabulary term `j` in the document. The result is usually a high-dimensional **sparse** matrix.

Minimal illustration:

```python
texts = ["i have a cat", "you have a dog", "you and i have a cat and a dog"]
vocabulary = list(enumerate(set(w for s in texts for w in s.split())))

def vectorize(tokens):
    return [sum(word == token for token in tokens) for _, word in vocabulary]
```

The article's example produces, for example, `[0, 1, 1, 1, 0, 0, 1]` for `"i have a cat"` (the order depends on the vocabulary). In production use `CountVectorizer`, sparse matrices, a bounded vocabulary, and stop-word handling rather than a hand-written implementation.

#### 4. N-grams: preserving local word order

Unigram BoW loses order: `"i have no cows"` and `"no, i have cows"` can receive the same vector despite different meanings. An **N-gram** is a sequence of `N` consecutive tokens. Include bigrams (or a range of n-gram sizes) to preserve local context:

```python
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(ngram_range=(1, 2))
X = vect.fit_transform(["no i have cows", "i have no cows"])
```

With `(1, 1)`, only `no`, `have`, and `cows` appear and both rows look identical. With `(1, 2)`, features such as `no have`, `have cows`, `have no`, and `no cows` distinguish the sentences. Larger N increases dimensionality and sparsity.

#### 5. Character N-grams

N-grams need not be words. Character N-grams (for example, 3-character windows with `analyzer="char_wb"`) capture similarity between related names and tolerate typos or spelling variants. In the article, character distances make `andersen` closer to `petersen` than to `smith`.

```python
vect = CountVectorizer(ngram_range=(3, 3), analyzer="char_wb")
X = vect.fit_transform(["andersen", "petersen", "petrov", "smith"])
```

#### 6. TF-IDF

BoW gives common words too much influence. **TF-IDF** increases the weight of a term that is frequent in a document but rare across the whole corpus.

For term `t`, document `d`, and corpus `D`, the article gives the smoothed inverse document frequency:

```text
idf(t, D) = log( |D| / (df(t) + 1) )
tfidf(t, d, D) = tf(t, d) * idf(t, D)
```

`tf(t, d)` is the term frequency and `df(t)` is the number of documents containing the term. Smoothing avoids division by zero and is the default-style formula shown in the article. TF-IDF is useful when domain-specific words separate classes from common words.

#### 7. Embeddings and the wider “bag” idea

BoW-style ideas generalize beyond text: a user session can be a **bag of sites**, and an app-usage record a **bag of apps**. These sparse methods are strong baselines.

Word embedding methods such as **Word2Vec**, **GloVe**, and **FastText** map words into dense vectors (often a few hundred dimensions). Similar contexts produce nearby vectors, enabling analogies such as:

```text
king - man + woman ~= queen
```

This is not literal understanding: Word2Vec arranges vectors from usage contexts. Meaningful semantic geometry normally requires very large training corpora, so pretrained models are often reused. The same idea can be applied in other domains (for example, “food2vec” or biological sequences).

### Images

Image feature extraction has two levels:

- **Hand-crafted features:** corners, edges, region borders, color statistics, brightness, and other domain-driven measurements. Libraries such as `skimage`, `SimpleCV`, and Pillow's `ImageStat` help. For rental photos, average pixel intensity can represent how bright an apartment appears.
- **Learned features:** convolutional neural networks (CNNs) learn hierarchical visual representations.

#### Pretrained CNNs and transfer learning

Instead of training a deep network from scratch, start with a pretrained model and its public weights. For **fine-tuning**, remove (“detach”) the final fully connected layers, add layers for the new task, and train on the new data. If only a vector representation is needed, remove the classifier head and use the preceding output:

```python
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
import numpy as np

resnet = ResNet50(include_top=False, weights="imagenet")
img = image.load_img("apartment.jpg", target_size=(224, 224))
x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
features = resnet.predict(x)
```

The extra batch dimension is required because the network expects a tensor shaped like `(batch_size, width, height, channels)`. Input size and model-specific preprocessing must match the pretrained network.

#### OCR and image metadata

If an image contains text, OCR can be more direct than a full neural model:

```python
import pytesseract
text = pytesseract.image_to_string(img)
```

`pytesseract` requires the Tesseract program and is not a universal solution; image quality, fonts, rotation, and layout can produce errors. The article demonstrates OCR on a simple image and on an apartment plan, where the output is imperfect.

Neural networks also cannot infer all useful non-visual information. Image **EXIF** metadata can provide camera manufacturer/model, resolution, flash use, GPS coordinates, and editing software. These fields are ordinary structured features and should be parsed separately.

### Geospatial data

Geographic inputs commonly appear as addresses or `(latitude, longitude)` coordinates.

#### Geocoding and reverse geocoding

- **Geocoding:** address -> coordinates.
- **Reverse geocoding:** coordinates -> address/nearby place.

Google Maps and OpenStreetMap expose APIs; `geopy` wraps several providers. At scale, API rate limits and HTTP latency make a local OpenStreetMap copy preferable. For small datasets, `reverse_geocoder` is a convenient local approximation:

```python
import reverse_geocoder as revgc

places = revgc.search(list(zip(df.latitude, df.longitude)))
```

#### Data quality and domain features

Addresses may contain typos. Coordinates can have GPS noise, low accuracy in tunnels or dense downtown areas, and Wi-Fi-based “teleportation” (a device in Manhattan may suddenly be assigned a router location in Chicago). Validate implausible jumps and remember that SSID/MAC mappings can be stale or duplicated.

Coordinates become useful when combined with domain knowledge. Candidate features include:

- distance to a subway station, store, ATM, or other landmark;
- building stories or elevation above sea level;
- local density and administrative area;
- distances along a route, not just point-to-point distance.

For connected points, calculate great-circle distance, road-network distance, number of turns, left/right-turn ratio, traffic lights, junctions, and bridges. A useful derived feature is **road complexity**:

```text
road_complexity = road-network distance / great-circle distance
```

### Date and time

Date/time fields need explicit feature design because calendars are cyclical and contain events.

#### Calendar features

Encode day of week as seven one-hot features and create a weekend indicator:

```python
df["dow"] = df["created"].apply(lambda x: x.date().weekday())
df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
```

Add task-specific calendar information: paydays, beginning/end of month, public holidays, major events, abnormal weather, sports events, or political ceremonies. These events can explain otherwise surprising anomalies.

#### Cyclic variables and harmonic encoding

Treating hour as an ordinary number creates an artificial boundary (`0` appears far from `23`), while one-hot encoding loses the notion that nearby hours are close. Project a periodic value onto a circle:

```python
import numpy as np

def make_harmonic_features(value, period=24):
    angle = value * 2 * np.pi / period
    return np.cos(angle), np.sin(angle)
```

This maps an hour to two coordinates and preserves cyclic proximity. It is useful for distance-based algorithms such as kNN, SVM, and k-means. For example, hours 23 and 1 have the same small circular distance as hours 9 and 11, while 9 and 21 are opposite (`distance = 2` in the article's normalized example). The article notes that alternative encodings may differ only slightly on a particular metric, so validate empirically.

### Time series, web, etc.

For time series, the article points to **tsfresh**, a library that automatically generates many descriptive features (lags, rolling statistics, and other time-series characteristics). Time ordering and leakage still need to be controlled when validating.

For web data, a User-Agent string contains several useful fields. Parse it rather than treating the raw string as one categorical value:

```python
import user_agents

ua = user_agents.parse(user_agent_string)
is_mobile = ua.is_mobile
os_family = ua.os.family
browser_family = ua.browser.family
browser_version = ua.browser.version
```

Also consider `is_bot`, `is_pc`, referrer, `http_accept_language`, and other headers. Combine browser version with the current latest version to form a “lag behind latest browser” feature.

An IP address may yield country, city, provider, and mobile/stationary connection type. IP databases can be outdated and proxies can add noise. Combining IP geography with language is informative: a Chilean proxy plus `ru_RU` browser locale may flag `is_traveler_or_proxy_user`. The general rule is to use intuition and domain knowledge to create features, then verify them against data quality and validation results.

---

## Feature transformations

Feature transformation changes existing values without necessarily adding new information. A monotonic transformation matters to some algorithms but has little or no effect on others. Decision trees, random forests, and gradient boosting are relatively robust to unusual distributions, which is one reason they are convenient baselines.

### Normalization and changing distribution

#### Why scale?

Parametric methods often work best when distributions are reasonably symmetric and unimodal. Distance-based methods such as kNN are especially sensitive to feature scale: a variable spanning hundreds of thousands can completely dominate one spanning `(-1, 1)`. Example: apartment distance from the city center may be measured in thousands of meters, while room count is usually below five.

There are also engineering reasons to use transformations: taking a logarithm can compress very large values and reduce numerical range. Scaling must be fitted on the training data only and then applied to validation/test data to avoid leakage.

#### Standard Scaling (Z-score)

```text
z = (x - mu) / sigma
```

where `mu` is the feature mean and `sigma` its standard deviation. In scikit-learn:

```python
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X_train)
```

Standard Scaling centers and rescales data; it **does not make a non-normal distribution normal**. The article demonstrates this with a Shapiro-Wilk test: the test statistic and tiny p-value remain essentially unchanged after scaling. Scaling can also be influenced by outliers, although it puts ordinary values on a comparable scale.

#### MinMax Scaling

```text
X_norm = (X - X_min) / (X_max - X_min)
```

```python
from sklearn.preprocessing import MinMaxScaler
X_scaled = MinMaxScaler().fit_transform(X_train)
```

This maps training values to a chosen interval, commonly `[0, 1]`. Standard and MinMax scaling often serve similar purposes. For distance calculations, Standard Scaling is usually the default; MinMax is useful when a fixed range is helpful for visualization (for example, `[0, 255]`). MinMax is particularly sensitive to extreme min/max values.

#### Log and other distribution-changing transforms

For a positive, right-skewed or log-normal variable, use a logarithm:

```python
price_log = np.log(price)
```

If `X` is log-normal, `log(X)` can be close to normal. The article's Shapiro-Wilk example changes from a very small p-value before the log to a non-rejecting p-value after it. The same idea can be tried for heavy right tails even when log-normality is only an approximation. If zeros or negative values exist, use `log(x + const)` with a justified constant, or consider:

- **Box-Cox** - includes logarithm as a special case but generally requires positive values.
- **Yeo-Johnson** - extends the idea to negative values.

Use a **Q-Q plot** as a visual diagnostic: a normal sample should approximately follow a smooth diagonal line. StandardScaler and MinMaxScaler change location/scale but not the Q-Q plot shape; taking the logarithm can make a strongly skewed price distribution look more normal. There is no universally best transformation, so compare cross-validated model quality.

### Interactions

Interactions encode relationships between variables that individual columns miss. In the rental example, total price is less informative than price per bedroom:

```python
rooms = df["bedrooms"].clip(lower=0.5)  # avoid division by zero
df["price_per_bedroom"] = df["price"] / rooms
```

The `0.5` floor is a practical safeguard, not a universal constant; choose it deliberately and inspect its effect. Do not generate every possible interaction without control: feature count can grow quickly, and many interactions have no domain meaning. Polynomial features are common in linear models but can become difficult to interpret. If the starting feature set is small, generate candidates and use feature selection/validation to remove unnecessary ones.

### Filling in the missing values

Most algorithms cannot accept missing values directly. Common tools include `pandas.DataFrame.fillna` and scikit-learn imputers. The article lists four straightforward strategies:

1. **Separate category:** encode missing categorical values as `"n/a"` or another explicit level.
2. **Typical value:** use mean/median for numeric features and the most frequent value for categorical features.
3. **Extreme sentinel:** use a deliberately unusual value, often useful with decision trees because the model can split missing from non-missing cases.
4. **Adjacent value:** for ordered data such as time series, carry forward or backward a neighboring observation.

Avoid blindly applying `df = df.fillna(0)`. Zero may have a real meaning and can hide an upstream data-collection bug. Fit imputation rules on training data and reuse them for validation/test data.

---

## Feature selection

Feature selection removes unimportant or harmful variables. It has two main motivations:

- **Computational cost:** more columns increase memory use, training time, inference latency, and operational complexity.
- **Generalization:** some algorithms interpret noise as signal and overfit, especially with many weak features.

Every selection method must be evaluated with the same cross-validation and target metric as the final model.

### Statistical approaches

The simplest candidate for removal is a constant feature: it contains no information. A softer extension is to remove features whose variance is below a threshold.

```python
from sklearn.feature_selection import VarianceThreshold

X_reduced = VarianceThreshold(threshold=0.9).fit_transform(X)
```

The article's synthetic matrix has shape `(100, 20)`:

- threshold `0.7` keeps all 20 features;
- threshold `0.8` keeps 18;
- threshold `0.9` keeps 12.

Variance is unsupervised: a low-variance feature can still be predictive, and a high-variance feature can still be noise. Other univariate statistical methods use a feature/target test, such as ANOVA F-test for classification:

```python
from sklearn.feature_selection import SelectKBest, f_classif

X_kbest = SelectKBest(f_classif, k=5).fit_transform(X, y)
```

On the article's artificial data, 5 best features improved mean 5-fold logistic-regression log loss from approximately `-0.311` to `-0.206` (scikit-learn reports negative log loss, so a value closer to zero is better). Variance filtering to 12 features gave approximately `-0.264`. The result is illustrative, not a guarantee; select the method and threshold with validation.

### Selection by modeling

Use a baseline model to estimate feature importance, then pass selected features to a final model. Two common choices are:

- **Random Forest:** tree ensembles provide importance estimates and can capture nonlinear structure.
- **Lasso-regularized linear model:** L1 regularization can shrink weak feature coefficients exactly to zero.

The intuition is: if a feature is useless to a simple, well-validated model, it is a questionable input to a more complex one. A scikit-learn pipeline keeps selection inside each training fold:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline

rf = RandomForestClassifier(n_estimators=10, random_state=17)
pipe = make_pipeline(SelectFromModel(estimator=rf), logit)
```

The article's synthetic example reports mean negative log loss around `-0.317` for logistic regression, `-0.241` for Random Forest, and `-0.220` for Random-Forest selection followed by logistic regression. In a second comparison with scaling, selection improved logistic regression (`-0.229`) over plain logistic regression (`-0.313`) and Random Forest (`-0.241`). These values depend on the random data and are not universal.

This approach is **not a silver bullet**: importance estimates can be biased, correlated features can hide one another, and discarding features can reduce performance. Always compare the selected pipeline with the unselected baseline.

### Grid search

The most direct and usually most reliable method is to evaluate feature subsets with the actual model and validation metric:

1. Choose a subset of columns.
2. Train and score the model.
3. Repeat for other subsets.
4. Keep the subset with the best cross-validated quality.

Trying every subset is **Exhaustive Feature Selection**. With `p` features, the number of subsets is exponential, so exhaustive search quickly becomes impractical.

#### Sequential Feature Selection

Reduce the search space by fixing a subset size `N`, testing combinations of `N`, keeping the best, and then adding one feature at a time until a maximum size or a quality plateau is reached. This is **forward sequential selection**. The reverse version starts with all features and removes one at a time while quality does not suffer; it is **backward selection**.

Example with `mlxtend` (backward selection, three final features):

```python
from mlxtend.feature_selection import SequentialFeatureSelector

selector = SequentialFeatureSelector(
    logit,
    scoring="neg_log_loss",
    k_features=3,
    forward=False,
    n_jobs=-1,
)
selector.fit(X, y)
```

Grid and sequential methods cost more computation than statistical filters or model-based heuristics, but they optimize the feature subset against the model and metric you actually care about. Use nested or carefully separated validation when the search is large to avoid selecting a subset that overfits the validation folds.

## Final checklist

- Start with domain understanding and clean raw inputs.
- Extract features appropriate to the data type (tokens, pixels, coordinates, time, metadata).
- Transform only when the algorithm or distribution benefits; scaling does not itself create normality.
- Encode cycles such as hour/day with calendar or harmonic features.
- Handle missingness according to the feature's meaning, not with an automatic zero.
- Keep interactions limited and protect against invalid arithmetic such as division by zero.
- Fit preprocessing and selection inside cross-validation pipelines to prevent leakage.
- Compare every “improvement” against a strong baseline using the final metric.
