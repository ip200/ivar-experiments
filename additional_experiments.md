# Experimental Improvements Plan (ICML Rebuttal Focus)

## 🎯 Goal

Strengthen the experimental section to directly address reviewer concerns while staying faithful to the theoretical claims of the paper.

---

## 🚨 Key Clarification

The reviewer’s suggestion to use **CCE (Conditional Calibration Error)** from van der Laan & Alaa is **not directly applicable**.

* Their CCE evaluates **coverage calibration of prediction intervals for Y**
* Our method produces **intervals for the conditional mean (or calibrated selector)**, *not label coverage*

👉 Therefore:

> We should **NOT report CCE as-is**, because it measures a different notion of validity.

---

## 🧠 Core Argument (for rebuttal)

> The CCE metric evaluates calibration of prediction intervals targeting label coverage, whereas our intervals target the conditional mean (or a selector calibrated for a transformed label). Since the target is not directly observable per test point, coverage-based calibration metrics are not applicable. We instead introduce metrics aligned with our theoretical guarantees.

---

## ✅ Proposed Experimental Additions

### 1. 📏 Interval Size Metrics (HIGH PRIORITY)

Directly address:

> “What about the size of the set predictions?”

Add:

* Mean width: `E[U - L]`
* Median width
* Std of width
* 90th percentile width

Optional:

* Normalize width by `std(y_train)` for comparability

---

### 2. 📊 Calibration Diagnostics (Point Predictor)

Since experiments already evaluate **merged point predictions (CVAR)**:

Add:

* Calibration slope & intercept:

  ```
  y ≈ a * prediction + b
  ```
* Bin-based calibration error:

  ```
  |E[Y | bin] - E[ŷ | bin]|
  ```
* Report:

  * Mean bin error
  * Max bin error

👉 This aligns with your existing experimental focus on point predictors.

---

### 3. 🔬 Synthetic Validity Experiment (STRONGEST ADDITION)

This is the **closest empirical validation of your theory**.

For synthetic datasets (e.g. Friedman):

#### Step:

* For each test point ( x ):

  * Estimate true ( E(Y | X = x) ) via Monte Carlo sampling

#### Report:

* Fraction of intervals containing the true conditional mean:

  ```
  coverage_mean = P( E[Y|X] ∈ [L, U] )
  ```
* Average width
* RMSE of merged predictor

👉 This directly matches your theoretical claim.

---

### 4. ⚖️ Width vs Accuracy Tradeoff

Add a table:

| Method | RMSE ↓ | Width ↓ |
| ------ | ------ | ------- |
| Base   | ...    | —       |
| IVAR   | ...    | ...     |
| CVAR   | ...    | ...     |

👉 Shows:

* whether intervals are informative
* whether better point predictions come at cost of width

---

### 5. 📈 Variance Reporting (EASY FIX)

Reviewer:

> “only means are reported”

Fix:

* Report **mean ± std** over seeds

Example:

```
3.124 ± 0.041
```

Optional:

* Add paired test (Wilcoxon / sign test)

---

### 6. 📊 Baseline Comparisons (Bounded vs Unbounded)

Add experiments on **bounded regression tasks**:

Compare:

* Base regressor
* Point calibration (isotonic / binning)
* **Bounded IVAR**
* **Unbounded IVAR/CVAR**

👉 Directly addresses reviewer request

---

### 7. 📄 Paper Structure Fix

Reviewer:

> “Include real datasets in main paper”

Action:

* Move 1–2 real datasets to main section
* Move synthetic details to appendix

---

### 8. ❓ Clarify “none” Option

Clarification:

> “none” = model trained on full training set, no calibration split

Add explicitly to paper.

---

## 🔧 Required Code Changes

### Fix dataset bug (IMPORTANT)

Friedman datasets are swapped:

```python
elif scenario == "friedman2":
    X, y = make_friedman2(...)
elif scenario == "friedman3":
    X, y = make_friedman3(...)
```

---

### Store per-sample outputs

Instead of only RMSE, store:

* predictions
* lower / upper bounds
* width
* residuals

---

### Add width metrics

```python
width = upper - lower

metrics = {
    "width_mean": width.mean(),
    "width_std": width.std(),
    "width_p90": np.percentile(width, 90),
}
```

---

### Add calibration metric

```python
def calibration_error(y_true, y_pred, n_bins=10):
    order = np.argsort(y_pred)
    bins = np.array_split(order, n_bins)

    errs = []
    for b in bins:
        if len(b) == 0:
            continue
        errs.append(abs(y_true[b].mean() - y_pred[b].mean()))

    return np.mean(errs)
```

---

## 📊 Suggested Final Table

| Method | RMSE ↓      | Calib Err ↓ | Width ↓ |
| ------ | ----------- | ----------- | ------- |
| Base   | 1.23 ± 0.04 | 0.08        | —       |
| IVAR   | 1.10 ± 0.03 | 0.04        | 0.32    |
| CVAR   | 1.05 ± 0.02 | 0.03        | 0.29    |

---

## 🧾 Rebuttal Summary (Copy-Paste Ready)

> We agree that the experimental section should include metrics more directly tied to the interval outputs. However, the conditional calibration error (CCE) used in prior work evaluates coverage calibration of prediction intervals for the label, whereas our method produces intervals targeting the conditional mean (or a calibrated selector). As this target is not directly observable per test point, coverage-based metrics are not applicable. Instead, we add interval width statistics, calibration diagnostics for the merged point predictor, and synthetic experiments measuring containment of the true conditional mean. We also include variance reporting and comparisons to bounded baselines.

---

## 🚀 Priority Order

1. Fix dataset bug
2. Add std to results
3. Add width metrics
4. Add calibration diagnostics
5. Add synthetic mean-containment experiment
6. Add bounded comparison

---

## ✅ Outcome

After these changes, you will have:

* answered all reviewer concerns
* stayed faithful to theory
* significantly strengthened experimental credibility

---

If you want, I can next:

* patch your `main.py` directly
* or generate the exact LaTeX tables for ICML
