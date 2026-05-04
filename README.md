# IVAR Benchmark Experiments

This repository contains the complete experimental suite for the **Inductive Venn-Abers (IVAR) Predictors for Regression** benchmark. It supports high-scale replication across 10 synthetic datasets and 4 real-world benchmarks, including automated LaTeX table generation.

## 🚀 Getting Started

### 1. Environment Setup
We recommend using a Python virtual environment (3.10+):
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
*Note: Key dependencies include `venn-abers==1.5.2`, `ucimlrepo`, `pandas`, `scikit-learn`, `scipy`, and `pypdf`.*

## 🧪 Running Experiments

### Individual Scenarios (`src/main.py`)
You can run a specific dataset configuration using the main entry point:

**Synthetic Example:**
```bash
python src/main.py --dataset synthetic_datasets --scenario linear_gaussian --n_samples 10000 --noise_level 3 --n_seeds 100 --save_details
```

**Real-World Example:**
```bash
python src/main.py --dataset real_datasets --scenario airfoil --n_seeds 100 --save_details
```

### Full Parallel Suite (`src/parallel_run.py`)
To replicate the entire 49-table suite efficiently, use the parallel runner. It detects your CPU cores and distributes the 100-seed jobs to maximize throughput.

```bash
python src/parallel_run.py
```
*   **Output:** Results are saved as individual CSVs in the `output/` directory.
*   **Scalability:** The script is optimized for Mac M-series or high-core workstations.

## 📊 Results & LaTeX Generation

### Automated Tables (`src/generate_tables.py`)
Once the experiments are complete, you can generate a comparison PDF that matches our experimental results.

```bash
python src/generate_tables.py
pdflatex -output-directory=output output/generate_tables.tex
```


## 📂 Project Structure
*   `src/main.py`: The core training/calibration loop.
*   `src/parallel_run.py`: Multi-core orchestration script.
*   `src/generate_tables.py`: LaTeX document generator.
*   `src/data/`: Data loading modules for UCI and local CSVs.
*   `output/`: Directory where all CSVs, .tex, and .pdf artifacts are stored.

---
