

# IVAR-experiments
A repository accompanying the paper Inductive Venn–Abers and related regressors containing code for running experiments on synthetic and real datasets and generating formatted result tables for analysis.

---

## Installation

Clone the repository and install the required Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Experiments

All experiments are executed from the `src` directory using the main module.

```bash
python -m main.py --dataset DATASET --noise_level NOISE_LEVEL
```

### Command-Line Arguments

- `--dataset`  
  Specifies the dataset type. Valid options are:
  - `synthetic_datasets`
  - `real_datasets`

- `--noise_level`  
  Specifies the noise level applied to the data. Valid options are:
  - `1`
  - `3`

  **Note:** This argument is only applicable when using `synthetic_datasets`. It is ignored when running experiments on real datasets.

---

### Example Usage

Run experiments on synthetic datasets with noise level 1:
```bash
python -m main.py --dataset synthetic_datasets --noise_level 1
```

Run experiments on real datasets:
```bash
python -m main.py --dataset real_datasets
```

---

## Output

All experiment outputs are saved to the `output/` subdirectory.  
This directory contains the raw results generated during execution.

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

