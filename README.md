# Inductive Venn–Abers and related regressors benchmark experiments

This repository contains the complete experimental suite for the **Inductive Venn–Abers and related regressors** publication. It supports high-scale replication across 11 synthetic datasets and 4 real-world benchmarks, including automated LaTeX table generation.

It also contains the two requested reviewer response experiments: **Loose bounds baseline** and **Self-Calibrating Conformal (SCC) comparison**.

---

## 🚀 Getting Started

### 1. Environment Setup
We recommend using a Python virtual environment (3.10+):
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install selfcalibratingconformal
```
*Note: Key dependencies include `venn-abers>=1.5.3` (which contains `VennAbersRegressor`), `ucimlrepo`, `pandas`, `scikit-learn`, `scipy`, and `pypdf`.*

---

## 🧪 Running Experiments

### 1. Individual Scenarios (`src/main.py`)
You can run a specific dataset configuration using the main entry point:

**Synthetic Example (incorporating Loose Bounds baseline):**
```bash
python src/main.py --dataset synthetic_datasets --scenario linear_gaussian --n_samples 10000 --noise_level 3 --n_seeds 100 --save_details
```
*The loose-bounds baseline (`CVAP - loose-bounds` and `CVAP - quantile-bounds`) is automatically executed as part of the main benchmark.*

**Real-World Example:**
```bash
python src/main.py --dataset real_datasets --scenario airfoil --n_seeds 100 --save_details
```

### 2. Full Parallel Suite (`src/parallel_run.py`)
To replicate the entire 49-table suite efficiently, use the parallel runner. It detects your CPU cores and distributes the 100-seed jobs to maximize throughput.
```bash
python src/parallel_run.py
```
*   **Output:** Results are saved as individual CSVs in the `output/` directory.
*   **Scalability:** The script is optimized for Mac M-series or high-core workstations.

### 3. Self-Calibrating Conformal Comparison (`run_scc_comparison.py`)
To compare CVAR against `selfcalibratingconformal` on Bounded Logistic, Linear Gaussian, and Heavy-tailed datasets:
```bash
python run_scc_comparison.py --n_seeds 100
```
*   **Output:** Results are saved as `output/scc_comparison_summary.csv` and `output/scc_comparison_details.csv`.

---

## 📊 Results & LaTeX Generation

### Automated Tables (`src/generate_tables.py`)
Once the experiments are complete, you can generate a comparison PDF that matches our experimental results.
```bash
python src/generate_tables.py
pdflatex -output-directory=output output/generate_tables.tex
```

---

## 📂 Project Structure

*   `src/main.py`: The core training/calibration loop (includes loose bounds).
*   `src/parallel_run.py`: Multi-core orchestration script.
*   `src/generate_tables.py`: LaTeX document generator.
*   `run_scc_comparison.py`: Conformal comparison runner.
*   `copa_followup.tex`: LaTeX file containing the reviewer response / paper text.
*   `src/data/`: Data loading modules for UCI and local CSVs.
*   `output/`: Directory where all CSVs, .tex, and .pdf artifacts are stored.

---

## 📝 Notes
- Ensure commands are executed from the correct directory.
- Noise levels are only relevant for synthetic datasets.
