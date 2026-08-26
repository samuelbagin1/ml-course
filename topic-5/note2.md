# Predicting Paying Users in a Mobile App — mlcourse.ai Lecture 5, Part 3

**Main point:** The model barely matters here — the real challenge is choosing the right metric. On imbalanced data (~7.5% payers), accuracy, ROC-AUC, and F-score mislead. Because the business only cares about the *top-K* users sent to Google's look-alike ad targeting, **lift** and **recall-at-fixed-precision** are what count.

## 1. The task
A mobile game wants to find likely payers, send their IDs to Google's look-alike API, and get back similar users to acquire.

```
 t0 ──────► t7/t8 ─────────────────► t90
 install   observe 8 days       did they pay?
           (launches, sessions,  (LTV @ day 90)
            payments)            = TARGET
```

- Observation window: first 8 days. Target: paid within 90 days (1 = paid).
- Paid in 8 days → definitely a payer at 90. But non-payers can still be predicted as future payers from behavior.
- Launch vs. session: one launch (app open) can hold several sessions (screen off / ~1 min idle starts a new one).
- It's like predicting **LTV**, except LTV is regression; here it's binary classification.
- Data: ~1M rows, ~50 cols; 32 features used (launches, sessions, payments per day).

## 2. Imbalance
```
Payers      ▓ 7.5%
Non-payers  ████████████████████████ 92.5%
```
A "nobody pays" model is already 92.5% accurate — so accuracy is a trap.

## 3. Time-aware cross-validation
Validation must sit in the future, with a 3-month gap to compute targets.

```
CORRECT: |<- TRAIN ->|<- 3mo gap ->|<- VALID ->|<- 3mo gap ->|
WRONG:   [train][valid][train][valid]   ✗ leaks future into past
```
Payments drift (promo spikes then decay), so future validation is the only honest test.

## 4. Models (minimal tuning)
- **Random Forest** — 300 trees, `class_weight='balanced'`, fixed seed, parallelized (~5 min).
- **Logistic Regression** — needs `StandardScaler` (fit on train, transform on valid), balanced weights, parallelizable solver.
> Note: the video ran logit on unscaled `X_valid` by mistake, hurting results; with scaling it actually does better.

## 5. Why the obvious metrics fail
**Accuracy** — misleading on imbalance:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
RF scored *worse* than a trivial baseline here.

**ROC-AUC** — ~96.5%, but "is that good?" is unanswerable, and it's computed over the whole ~350K tail while the business only cares about the top of the ranking.
$$\text{ROC-AUC} = 1 - P(\text{random negative ranked above random positive})$$

## 6. The metric that matters: Lift
Send the **top 50,000** likely payers to Google. Measure payer rate in that slice vs. base rate:
$$\text{Lift@K} = \frac{\text{payer rate in top-}K}{\text{payer rate overall}}$$

| Selection | Payer rate (top 50K) | Overall | Lift |
|---|---|---|---|
| Random Forest | ~43% | ~7% | **~6.3** |
| Baseline (paid in 8d) | ~40% | ~7% | **~5.87** |

Trusting the model's top-50K is ~6.3× better than sending random users.

## 7. Recall at fixed precision
Business rule: "at least half of who we send must really pay." So maximize recall subject to precision ≥ 0.5:
$$\text{Precision} = \frac{TP}{TP+FP} \qquad \text{Recall} = \frac{TP}{TP+FN}$$

Can't just max recall (threshold τ=0 predicts everyone pays). Tune τ from the precision-recall curve:

```
metric
1.0 │██──────────────────╱ precision
0.5 │┈┈┈╲┈┈┈┈┈┈┈┈┈┈┈┈╱┈┈┈ min precision
    │    ╲___     ╱  recall
0.0 │__________╳__________
    0      τ*=0.184       1
```
Optimal **τ ≈ 0.184**: precision just above 0.5, recall maximized.

## 8. Confusion matrix
$$
\begin{array}{c|cc}
 & \text{Pred 0} & \text{Pred 1} \\
\hline
\text{True 0} & TN & FP \\
\text{True 1} & FN & TP
\end{array}
$$
- Baseline: precision = 1.0 (no false positives) but more false negatives.
- Tuned RF: more false positives (OK, precision still ≥ 0.5), far fewer false negatives.
- Result: baseline finds ~20,000 payers; tuned RF finds **~21,000+**.

## 9. Feature importance (RF)
1. Payments day 8 (top) → 2. payments day 7, day 6 → 3. sessions & launches day 8.
Payments dominate, but sessions/launches help flag payers who hadn't paid yet.

## 10. Where the real gains are
Beyond launches/sessions/payments: demographics (country, region, age, gender → bag-of-regions) and especially **in-game events** (logins, purchases, level-ups — thousands of types → bag-of-events counts). These are sparse, which is why Logistic Regression is worth trying.

## 11. Verdict
RF chosen — but if RF works, tuned **gradient boosting** is usually better/faster; use **PCA** for very high dimensionality. LR vs. trees is a real tradeoff.

## TL;DR
1. Window = 8 days, target = paid by day 90.
2. Handle imbalance with balanced weights.
3. Time-aware CV (validation in future, 3-mo gap).
4. Train RF (+ scaled LR).
5. Ignore accuracy/F-score/raw AUC → use **lift@50K**.
6. Tune τ ≈ 0.184 for recall at precision ≥ 0.5.
7. Read confusion matrix + feature importances.
8. Add in-game events & geo for production gains.

*Source: Yury Kashnitsky, "mlcourse.ai. Lecture 5. Part 3" (2018) · https://mlcourse.ai · https://github.com/Yorko/mlcourse.ai*
