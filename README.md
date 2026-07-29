

# IVAR-experiments
A repository accompanying the paper Inductive Venn–Abers and related regressors containing code for running experiments on synthetic and real datasets and generating formatted result tables for analysis.

---

## Installation

Clone the repository and install the required Python dependencies:

```bash
pip install -r requirements.txt
pip install selfcalibratingconformal
```

---

## Running Experiments

All experiments are executed from the `src` directory or root directory using the main module.

### 1. Main Benchmark (with Bounded-IVAR/Loose-Bounds baseline)

The loose-bounds baseline (`CVAP - loose-bounds` and `CVAP - quantile-bounds`) is automatically run as part of the main benchmark script.

```bash
python -m main.py --dataset DATASET --noise_level NOISE_LEVEL
```

#### Command-Line Arguments

- `--dataset`  
  Specifies the dataset type. Valid options are:
  - `synthetic_datasets`
  - `real_datasets`

- `--noise_level`  
  Specifies the noise level applied to the data. Valid options are:
  - `1`
  - `3`

  **Note:** This argument is only applicable when using `synthetic_datasets`. It is ignored when running experiments on real datasets.

- `--n_seeds`  
  Number of seeds to run (default: 10).

- `--save_details`  
  Saves per-sample predictions.

### 2. SelfCalibratingConformal Comparison

To compare CVAR against `selfcalibratingconformal` on Bounded Logistic, Linear Gaussian, and Heavy-tailed datasets:

```bash
python run_scc_comparison.py --n_seeds 100
```

---

### Example Usage

Run experiments on synthetic datasets with noise level 3, 10000 samples, and 10 seeds:
```bash
python src/main.py --dataset synthetic_datasets --noise_level 3 --n_samples 10000 --n_seeds 10
```

Run SCC comparison with 100 seeds:
```bash
python run_scc_comparison.py --n_seeds 100
```

---

## Output

All experiment outputs are saved to the `output/` subdirectory.  
This directory contains the raw results generated during execution:
- `scc_comparison_summary.csv` and `scc_comparison_details.csv` for the conformal comparison.
- `synthetic_datasets_noise_*_*.csv` for the main benchmark.

---

## Results Processing

To process the experimental results and generate tex formatted tables, run the following Jupyter notebook:

```text
process_results.ipynb
```

The notebook reads data from the `output/` directory and produces **text-formatted tables** suitable for reporting and analysis.

---

## Notes

- Ensure commands are executed from the correct directory as specified above.
- Noise levels are only relevant for synthetic datasets.
- The repository is structured to support reproducible experimentation.

---

