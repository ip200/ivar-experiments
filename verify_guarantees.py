import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('src'))

from main import run_benchmark

# Run for a standard scenario with more samples and seeds to get a stable estimate
scenarios = ['linear_gaussian']
df, summary = run_benchmark(scenarios, seeds=range(5), n_samples=1000, noise_scale=1.0)

# Print mean_containment for all CVAP models
cvap_summary = summary[summary['model'].str.contains('CVAP')]
print("\nMean Containment Summary (linear_gaussian):")
cols = ['model', 'mean_containment_mean', 'width_mean_mean']
print(cvap_summary[cols])

# Run for bounded_logistic
scenarios = ['bounded_logistic']
df_b, summary_b = run_benchmark(scenarios, seeds=range(5), n_samples=1000, noise_scale=1.0)
cvap_summary_b = summary_b[summary_b['model'].str.contains('CVAP')]
print("\nMean Containment Summary (bounded_logistic):")
print(cvap_summary_b[cols])
