import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('src'))

from main import run_benchmark, dataset_dict

scenarios = ['linear_gaussian', 'bounded_logistic']
df, summary = run_benchmark(scenarios, seeds=[0], n_samples=500, noise_scale=1.0)

# Filter for a CVAP model
cvap_rows = summary[summary['model'].str.contains('CVAP')]
if not cvap_rows.empty:
    print("\nFirst CVAP row in summary:")
    print(cvap_rows.iloc[0])
else:
    print("\nNo CVAP rows found in output!")

# Check mean_containment for bounded_logistic
bounded_cvap = summary[(summary['scenario'] == 'bounded_logistic') & (summary['model'].str.contains('CVAP'))]
if not bounded_cvap.empty:
    print("\nBounded CVAP row in summary:")
    print(bounded_cvap.iloc[0][['scenario', 'model', 'mean_containment_mean', 'width_mean_mean']])
