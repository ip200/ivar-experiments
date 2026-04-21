# ICML Rebuttal Handover: Experimental Improvements

This document summarizes the changes made to the `ivar-experiments` repository to address reviewer feedback and provides instructions for running the updated experiment suite on a new machine.

## 🛠 Summary of Changes

### 1. Code Infrastructure Improvements
*   **Enhanced Dataset Support**:
    *   Added support for the **Wine Quality (Red)** dataset in `src/data/uci_repository_datasets/datasets.py`.
    *   Fixed a bug where `friedman2` and `friedman3` were swapped in experiments.
    *   Updated type hints to be compatible with Python 3.9.
*   **Pipeline Refactoring**:
    *   Updated `src/main.py` to calculate **Mean Containment** (fraction of intervals containing the true conditional mean).
    *   Implemented **Width Normalization**: All interval widths are now normalized by `std(y_train)` for cross-dataset comparability.
    *   Implemented **Calibration Diagnostics**: Added bin-based calibration error for point predictors.
    *   **Interval Extraction Fix**: Corrected the interval handling logic to correctly match the `VennAberRegressor` output format (`[L1...LN, U1...UN]`).

### 2. New Experimental Scenarios
*   **Bounded Regression**: Added the `wine_red` dataset (real-world) and a new `bounded_logistic` synthetic scenario (0–10 scale) to address reviewer requests.
*   **CLI Enhancements**: Added `--n_seeds` and `--save_details` arguments to `src/main.py` for more controlled and detailed experimentation.

### 3. Diagnostic & Proof Scripts
The following new scripts were added to the root directory:
*   `proof_of_dependence.py`: Demonstrates that interval quality (containment) is strongly correlated with the underlying model's ability to capture the signal.
*   `diagnostic_lasso.py`: A focused comparison showing how Lasso's regularization strength affects containment.
*   `verify_guarantees.py`: A validation script checking `mean_containment` against theoretical expectations.

---

## 🚀 How to Run the Experiments

### 1. Environment Setup
Ensure you are in the project root and have Python 3.9+ installed.

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install matplotlib  # Required for diagnostic plots
```

### 2. Running Main Benchmark
To run the full synthetic benchmark with the new metrics and detailed output:

```bash
mkdir -p output
python3 src/main.py --dataset synthetic_datasets --n_samples 10000 --noise_level 1 --n_seeds 10 --save_details
```

### 3. Running Diagnostic Proofs
To generate the "Experimental Proof" (Correlation: -0.99) between model accuracy and containment:

```bash
python3 proof_of_dependence.py
```
*This will generate `lasso_proof.png` and a detailed results table.*

To check how Lasso bias impacts results specifically:
```bash
python3 diagnostic_lasso.py
```

---

## 📅 Next Steps for the Rebuttal
Use the generated CSVs in `output/` to:
1.  **Report Interval Widths**: Use the `width_mean_mean` and `width_std_mean` columns.
2.  **Validate Guarantees**: Use the `mean_containment_mean` column (expect values $\approx 95\%$ for $m=1$).
3.  **Address Bounded Regression**: Compare `wine_red` and `bounded_logistic` results against baselines.
