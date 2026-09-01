import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# Add src to path
sys.path.append(os.path.abspath('src'))

from main import make_synth_regression, compute_metrics
from venn_abers import VennAbersRegressor

def run_lasso_sweep():
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
    
    alphas = np.logspace(-4, 0.5, 30) # Sweep from small to large
    results = []
    
    true_w = ds.meta['w']
    
    for alpha in alphas:
        model = Lasso(alpha=alpha, random_state=seed)
        model.fit(ds.X_train, ds.y_train)
        
        # Weight MSE
        fitted_w = model.coef_
        weight_mse = np.mean((true_w - fitted_w)**2)
        
        # Venn-Abers
        va = VennAbersRegressor(estimator=model, inductive=False, n_splits=5, random_state=seed)
        va.fit(ds.X_train, ds.y_train, m=1)
        va_preds, intervals = va.predict(ds.X_test)
        
        metrics = compute_metrics(
            ds.y_test, va_preds, intervals=intervals, 
            y_true_mean=ds.y_true_mean, y_train=ds.y_train,
            model=model, true_w=true_w
        )
        
        results.append({
            'alpha': alpha,
            'rmse': metrics['rmse'],
            'weight_mse': weight_mse,
            'mean_containment': metrics['mean_containment'],
            'width_mean': metrics['width_mean']
        })
        print(f"Alpha: {alpha:.4f}, RMSE: {metrics['rmse']:.4f}, Weight MSE: {weight_mse:.4f}, Containment: {metrics['mean_containment']:.4f}")
        
    res_df = pd.DataFrame(results)
    res_df.to_csv('output/lasso_diagnostic_sweep.csv', index=False)
    
    # Plotting
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(res_df['rmse'], res_df['mean_containment'], alpha=0.8, s=100, edgecolors='k')
    plt.title('Lasso Diagnosis: RMSE vs Mean Containment', fontsize=14)
    plt.xlabel('RMSE (Point Predictor Error)', fontsize=12)
    plt.ylabel('Mean Containment ($E[Y|X]$)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.scatter(res_df['weight_mse'], res_df['mean_containment'], alpha=0.8, s=100, edgecolors='k')
    plt.title('Lasso Diagnosis: Weight MSE vs Mean Containment', fontsize=14)
    plt.xlabel('Weight MSE (Coefficient Recovery Error)', fontsize=12)
    plt.ylabel('Mean Containment ($E[Y|X]$)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('output/lasso_diagnostic_plots.png', dpi=300)
    print("Plots saved to output/lasso_diagnostic_plots.png")

if __name__ == "__main__":
    run_lasso_sweep()
