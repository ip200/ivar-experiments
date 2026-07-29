import numpy as np
import pandas as pd
import argparse
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from selfcalibratingconformal import SelfCalibratingConformalPredictor
from venn_abers import VennAbersRegressor
import warnings
import sys
import os

# Ensure src is in sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.main import make_synth_regression, compute_metrics

warnings.filterwarnings("ignore")

def run_scc_experiment(scenarios, seeds, n_samples=10000, noise_scale=3.0):
    rows = []
    
    for sc in scenarios:
        print(f"Running scenario: {sc}")
        for seed in seeds:
            # Generate dataset
            ds = make_synth_regression(
                n_samples=n_samples,
                n_features=10,
                scenario=sc,
                noise_scale=noise_scale,
                random_state=seed,
            )
            
            # 1. Base model trained on full 80% train data
            base_model = GradientBoostingRegressor(random_state=seed)
            base_model.fit(ds.X_train, ds.y_train)
            base_preds = base_model.predict(ds.X_test)
            base_metrics = compute_metrics(ds.y_test, base_preds, y_train=ds.y_train)
            
            rows.append({
                "scenario": sc,
                "seed": seed,
                "method": "Base",
                "rmse_y": base_metrics.get("rmse"),
                "rmse_mean": base_metrics.get("rmse") if ds.y_true_mean is None else np.sqrt(np.mean((ds.y_true_mean - base_preds) ** 2)),
                "calib_err": base_metrics.get("calib_err"),
                "coverage": None,
                "width": None
            })
            
            # 2. CVAR1 (CVAP - 1) on full 80% train data
            va = VennAbersRegressor(estimator=GradientBoostingRegressor(random_state=seed), inductive=False, n_splits=10, random_state=seed)
            va.fit(ds.X_train, ds.y_train, m=1)
            va_preds, intervals = va.predict(ds.X_test)
            n_samples_test = len(va_preds)
            lower = intervals[:n_samples_test]
            upper = intervals[n_samples_test:]
            cvar_intervals = np.column_stack((lower, upper))
            cvar_metrics = compute_metrics(
                ds.y_test, va_preds, intervals=cvar_intervals, y_true_mean=ds.y_true_mean, y_train=ds.y_train
            )
            
            rows.append({
                "scenario": sc,
                "seed": seed,
                "method": "CVAR1",
                "rmse_y": cvar_metrics.get("rmse"),
                "rmse_mean": np.sqrt(np.mean((ds.y_true_mean - va_preds) ** 2)) if ds.y_true_mean is not None else None,
                "calib_err": cvar_metrics.get("calib_err"),
                "coverage": None,
                "width": cvar_metrics.get("width_mean")
            })
            
            # 3. SCC - split train into 75% proper train and 25% calibration
            X_proper, X_cal, y_proper, y_cal = train_test_split(
                ds.X_train, ds.y_train, test_size=0.25, random_state=seed
            )
            
            scc_base = GradientBoostingRegressor(random_state=seed)
            scc_base.fit(X_proper, y_proper)
            
            predictor = lambda X: scc_base.predict(X)
            scc = SelfCalibratingConformalPredictor(predictor=predictor)
            scc.calibrate(X_cal, y_cal, alpha=0.05) # targets 95% coverage
            
            # SCC-point
            scc_point_preds = scc.predict_point(ds.X_test)
            scc_point_metrics = compute_metrics(ds.y_test, scc_point_preds, y_train=ds.y_train)
            
            rows.append({
                "scenario": sc,
                "seed": seed,
                "method": "SCC-point",
                "rmse_y": scc_point_metrics.get("rmse"),
                "rmse_mean": np.sqrt(np.mean((ds.y_true_mean - scc_point_preds) ** 2)) if ds.y_true_mean is not None else None,
                "calib_err": scc_point_metrics.get("calib_err"),
                "coverage": None,
                "width": None
            })
            
            # SCC-interval
            scc_intervals = scc.predict_interval(ds.X_test)
            # Calculate coverage of Y
            scc_coverage = np.mean((ds.y_test >= scc_intervals[:, 0]) & (ds.y_test <= scc_intervals[:, 1]))
            # Width normalized by std(y_train) for consistency
            norm_factor = np.std(ds.y_train) if np.std(ds.y_train) > 0 else 1.0
            scc_widths = (scc_intervals[:, 1] - scc_intervals[:, 0]) / norm_factor
            scc_mean_width = np.mean(scc_widths)
            
            # For SCC-interval point predictions, we can use the interval midpoint
            scc_midpoints = (scc_intervals[:, 0] + scc_intervals[:, 1]) / 2.0
            scc_mid_rmse_y = np.sqrt(np.mean((ds.y_test - scc_midpoints) ** 2))
            scc_mid_rmse_mean = np.sqrt(np.mean((ds.y_true_mean - scc_midpoints) ** 2)) if ds.y_true_mean is not None else None
            
            rows.append({
                "scenario": sc,
                "seed": seed,
                "method": "SCC-interval",
                "rmse_y": scc_mid_rmse_y,
                "rmse_mean": scc_mid_rmse_mean,
                "calib_err": None,
                "coverage": scc_coverage,
                "width": scc_mean_width
            })
            
    df = pd.DataFrame(rows)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=100)
    args = parser.parse_args()
    
    scenarios = ["bounded_logistic", "linear_gaussian", "heavy_tailed"]
    seeds = range(args.n_seeds)
    
    print(f"Running SCC Comparison Experiment on {args.n_seeds} seeds...")
    df = run_scc_experiment(scenarios, seeds)
    
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/scc_comparison_details.csv", index=False)
    
    # Aggregation
    summary = df.groupby(["scenario", "method"]).agg({
        "rmse_y": ["mean", "std"],
        "rmse_mean": ["mean", "std"],
        "calib_err": ["mean", "std"],
        "coverage": ["mean", "std"],
        "width": ["mean", "std"]
    })
    
    print("\n" + "="*80)
    print("SCC COMPARISON SUMMARY")
    print("="*80)
    print(summary)
    
    summary.to_csv("output/scc_comparison_summary.csv")

if __name__ == "__main__":
    main()
