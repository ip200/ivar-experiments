import numpy as np
import pandas as pd
import sys
import os
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error

# Add src to path
sys.path.append(os.path.abspath('src'))

from main import make_synth_regression, compute_metrics
from venn_abers import VennAberRegressor

# Run for a standard scenario
scenario = 'linear_gaussian'
n_samples = 1000
seed = 0
noise_scale = 1.0

ds = make_synth_regression(
    n_samples=n_samples,
    n_features=10,
    scenario=scenario,
    noise_scale=noise_scale,
    random_state=seed,
)

def run_diagnostic(name, model_class, **kwargs):
    model = model_class(**kwargs)
    model.fit(ds.X_train, ds.y_train)
    
    # Coefficients comparison
    if hasattr(model, 'coef_'):
        true_w = ds.meta['w']
        fitted_w = model.coef_
        print(f"\n[{name}] Coefficients comparison:")
        print(f"True w (first 5):   {true_w[:5]}")
        print(f"Fitted w (first 5): {fitted_w[:5]}")
        print(f"MSE of coefficients: {np.mean((true_w - fitted_w)**2):.4f}")

    # Point prediction performance
    y_pred = model.predict(ds.X_test)
    print(f"[{name}] RMSE on test: {np.sqrt(mean_squared_error(ds.y_test, y_pred)):.4f}")

    # Venn-Abers calibration
    va = VennAberRegressor(estimator=model, inductive=False, n_splits=10, random_state=seed)
    va.fit(ds.X_train, ds.y_train, m=1)
    va_preds, intervals = va.predict(ds.X_test)
    
    n = len(va_preds)
    lower = intervals[:n]
    upper = intervals[n:]
    intervals_stacked = np.column_stack((lower, upper))
    
    metrics = compute_metrics(ds.y_test, va_preds, intervals=intervals_stacked, y_true_mean=ds.y_true_mean, y_train=ds.y_train)
    print(f"[{name}] VA Mean Containment: {metrics['mean_containment']:.4f}")
    print(f"[{name}] VA Mean Width: {metrics['width_mean']:.4f}")

run_diagnostic("LinearRegression", LinearRegression)
run_diagnostic("Lasso", Lasso, alpha=1.0) # Default alpha is 1.0
run_diagnostic("Lasso (Small Alpha)", Lasso, alpha=0.01)
