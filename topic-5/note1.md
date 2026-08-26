# Classification Metrics — and Why ROC AUC Is the Default for Probability Models

**The main point:** No single classification metric is "correct." The right one depends on your business task — specifically, whether a false positive or a false negative hurts more. Accuracy hides problems on imbalanced data, precision and recall each cover one type of error, and **ROC AUC** is the go-to when your model outputs probabilities instead of hard labels, because it measures how well the model *ranks* positives above negatives across every possible threshold.

---

## The Confusion Matrix

Everything starts here. Predictions vs. true labels give four outcomes:

|                          | Actual Positive (1) | Actual Negative (0) |
| ------------------------ | ------------------- | ------------------- |
| **Predicted Positive (1)** | True Positive (TP)  | False Positive (FP) |
| **Predicted Negative (0)** | False Negative (FN) | True Negative (TN)  |

- **False Positive** = model says positive, reality is negative ("you're pregnant" → to a man).
- **False Negative** = model says negative, reality is positive ("you're not pregnant" → to a very pregnant woman).

---

## Accuracy — Simple, but Deceptive

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Everyone understands it — it's just the share of correct answers. But it has two serious weaknesses:

1. **It treats FP and FN equally.** They carry the same coefficient in the formula. Yet the cost differs wildly by task:
   - **Spam detection** → a **false positive** is worse (your boss's email gets deleted).
   - **Cancer screening** → a **false negative** is worse (you miss a sick patient).
   - **Search results** → genuinely ambiguous, you need a weighted blend.

2. **It collapses on imbalanced data.** If only 1% of users pay, a model that predicts "nobody pays" scores **99% accuracy** while doing nothing useful. Always compare against the naive baseline:

$$\text{baseline accuracy} = 1 - \bar{y}$$

where $\bar{y}$ is the mean target (the positive rate).

---

## Precision and Recall — One Error Each

When you care about one error type specifically:

$$\text{Precision} = \frac{TP}{TP + FP} \qquad \text{(controls false positives)}$$

$$\text{Recall} = \frac{TP}{TP + FN} \qquad \text{(controls false negatives)}$$

- **Precision** looks at the top row — of everything you *flagged* positive, how much was right? Use it when FP is costly (spam).
- **Recall** looks at the left column — of everything *truly* positive, how much did you catch? Use it when FN is costly (cancer, churn, anomaly detection).

You usually can't max both at once, so combine them with the **F1 score** (harmonic mean):

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

F1 is the common replacement for accuracy on imbalanced problems.

> Note: the lecturer mixes up micro- vs. macro-averaging around the 26-minute mark — he later pinned a correction. In practice he says the averaging choice matters less than actually *reading* the confusion matrix to see which classes get confused.

---

## ROC AUC — The Default for Probability Outputs

Often your model outputs a **score or probability**, not a 0/1 label. You sort all instances by predicted probability (a "scoring" or scorecard), then decide a threshold later. ROC AUC evaluates the *ranking* without committing to any single threshold.

### The two axes

ROC plots two rates as you sweep the threshold:

$$\text{TPR (True Positive Rate)} = \frac{TP}{TP + FN} \quad (= \text{Recall})$$

$$\text{FPR (False Positive Rate)} = \frac{FP}{FP + TN}$$

### How the curve is built

Pick a threshold $\tau$; everything above it is predicted positive. Each threshold produces one (FPR, TPR) point. Sweep $\tau$ from 0 to 1 and connect the dots:

- $\tau = 0$ → predict everything positive → **TPR = 1, FPR = 1** (top-right corner).
- $\tau = 1$ → predict everything negative → **TPR = 0, FPR = 0** (bottom-left corner).
- Worked example (6 objects): at threshold 0.13 the model got **TPR = 2/3** and **FPR = 1/3**.

```
TPR
 1 |        ┌────────●  ← ideal point (FPR=0, TPR=1)
   |     ┌──┘
   |   ┌─┘        ····  ← random model (diagonal, AUC = 0.5)
   | ┌─┘      ····
   |─┘    ····
 0 └───────────────── FPR
   0                1
```

**AUC = the area under this curve.**

### Interpreting the number

| AUC             | Meaning                                       |
| --------------- | --------------------------------------------- |
| **1.0**         | Perfect — curve hugs the top-left corner      |
| **0.5**         | Random guessing — the diagonal line           |
| **< 0.5**       | Worse than random (no reason to look here)    |
| worked example  | **8/9 ≈ 0.89**                                |

### The key insight — AUC is a ranking metric

AUC equals the **share of correctly ranked positive–negative pairs**. A pair is "wrong" only when a negative example gets a higher score than a positive one.

$$\text{AUC} = \frac{\text{correctly ranked }(+,-)\text{ pairs}}{\text{total }(+,-)\text{ pairs}}$$

In the 6-object example: 3 positives × 3 negatives = **9 pairs**, exactly **1 misranked** → $\frac{9-1}{9} = \frac{8}{9}$. This matches the geometric area exactly.

Two consequences:

- Because it's rank-based, **multiplying all predictions by a constant doesn't change AUC** — it's scale-invariant, so it doesn't care whether outputs are true probabilities.
- **AUC is poorly interpretable for business decisions.** "AUC = 0.95" only tells you "closer to 1 is better" — it doesn't map to money or actions. Great for Kaggle (fixed metric), weaker for real projects.

---

## Lift — The Business-Friendly Alternative

When you can only act on the top-K ranked instances (e.g., call 100 churn-risk customers a day), **lift** is more actionable. It blends ROC's ranking idea with accuracy's simplicity:

$$\text{Lift@K} = \frac{\text{positive rate in top } K}{\text{positive rate overall } (\bar{y})}$$

A lift of 4 means: targeting your model's top picks finds **4× more** churners than random targeting — so your outreach budget is 4× more efficient. K is usually set by business constraints (often ~10%, or however many people you can actually contact).

---

## The Takeaway

Pick your metric *before* training, and pick it for the task:

- **Balanced, symmetric costs** → Accuracy is fine.
- **Care about false positives** → Precision. **Care about false negatives** → Recall. **Both** → F1.
- **Model outputs probabilities / you want threshold-free ranking quality** → **ROC AUC**.
- **Acting on a top-K shortlist with a real budget** → Lift.

*Source: "mlcourse.ai. Lecture 5. Part 2. Classification metrics. Theory" by Yury Kashnitsky.*
