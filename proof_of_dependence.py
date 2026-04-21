import numpy as np
import pandas as pd
import sys
import os
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath('src'))

from main import make_synth_regression, compute_metrics
from venn_abers import VennAberRegressor

# Setup
scenario = 'linear_gaussian'
n_samples = 2000 # More samples for stability
seed = 42
noise_scale = 1.0

ds = make_synth_regression(
    n_samples=n_samples,
    n_features=10,
    scenario=scenario,
    noise_scale=noise_scale,
    random_state=seed,
)

alphas = np.logspace(-4, 1, 10)
results = []

for alpha in alphas:
    model = Lasso(alpha=alpha)
    model.fit(ds.X_train, ds.y_train)
    
    y_pred = model.predict(ds.X_test)
    rmse_val = np.sqrt(np.mean((ds.y_test - y_pred)**2))
    
    # Coefficients MSE
    true_w = ds.meta['w']
    w_mse = np.mean((true_w - model.coef_)**2)
    
    va = VennAberRegressor(estimator=model, inductive=False, n_splits=5, random_state=seed)
    va.fit(ds.X_train, ds.y_train, m=1)
    va_preds, intervals = va.predict(ds.X_test)
    
    n = len(va_preds)
    lower = intervals[:n]
    upper = intervals[n:]
    intervals_stacked = np.column_stack((lower, upper))
    
    metrics = compute_metrics(ds.y_test, va_preds, intervals=intervals_stacked, y_true_mean=ds.y_true_mean, y_train=ds.y_train)
    
    results.append({
        "alpha": alpha,
        "model_rmse": rmse_val,
        "weight_mse": w_mse,
        "mean_containment": metrics['mean_containment'],
        "width": metrics['width_mean']
    })

df_res = pd.DataFrame(results)
print("\n--- Experimental Proof Table ---")
print(df_res[['alpha', 'model_rmse', 'weight_mse', 'mean_containment', 'width']])

# Simple correlation check
corr = df_res['model_rmse'].corr(df_res['mean_containment'])
print(f"\nCorrelation between Model RMSE and Mean Containment: {corr:.4f}")

# Plotting
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.semilogx(df_res['alpha'], df_res['mean_containment'], marker='o')
plt.xlabel('Lasso Alpha (Regularization Strength)')
plt.ylabel('Mean Containment')
plt.title('Containment vs Regularization')

plt.subplot(1, 2, 2)
plt.scatter(df_res['model_rmse'], df_res['mean_containment'])
plt.xlabel('Model RMSE (Error)')
plt.ylabel('Mean Containment')
plt.title('Containment vs Model Error')
plt.tight_layout()
plt.savefig('lasso_proof.png')
print("\nPlot saved as lasso_proof.png")
