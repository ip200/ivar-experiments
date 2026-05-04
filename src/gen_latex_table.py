import pandas as pd
import numpy as np
from scipy import stats

def generate_full_synthetic_table():
    summary_df = pd.read_csv("output/synthetic_datasets_noise_1_10000.csv")
    details_df = pd.read_csv("output/synthetic_datasets_noise_1_10000_details.csv")
    
    scenarios = summary_df['scenario'].unique()
    regressors = ["LinearRegression", "Ridge", "Lasso", "ElasticNet", "RandomForest", "GradientBoosting", "SVR(RBF)"]
    regressor_names = {
        "LinearRegression": "Linear Regression",
        "Ridge": "Ridge",
        "Lasso": "Lasso",
        "ElasticNet": "ElasticNet",
        "RandomForest": "Random Forest",
        "GradientBoosting": "Gradient Boosting",
        "SVR(RBF)": "SVR (RBF)"
    }
    
    latex_rows = []
    
    all_base_rmse = []
    all_cvap1_rmse = []
    all_cvap10_rmse = []

    for scenario in scenarios:
        latex_rows.append(f"\\midrule\n\\multicolumn{{4}}{{c}}{{\\textbf{{Scenario: {scenario.replace('_', ' ').title()}}}}} \\\\ \\midrule")
        
        scenario_base = []
        scenario_cvap1 = []
        scenario_cvap10 = []

        for reg in regressors:
            base_row = summary_df[(summary_df['scenario'] == scenario) & (summary_df['model'] == reg)]
            cvap1_row = summary_df[(summary_df['scenario'] == scenario) & (summary_df['model'] == f"{reg} CVAP - 1")]
            cvap10_row = summary_df[(summary_df['scenario'] == scenario) & (summary_df['model'] == f"{reg} CVAP - 10")]
            
            if base_row.empty or cvap1_row.empty or cvap10_row.empty:
                continue
                
            base_rmse = base_row['rmse_mean'].values[0]
            base_std = base_row['rmse_std'].values[0]
            cvap1_rmse = cvap1_row['rmse_mean'].values[0]
            cvap1_std = cvap1_row['rmse_std'].values[0]
            cvap10_rmse = cvap10_row['rmse_mean'].values[0]
            cvap10_std = cvap10_row['rmse_std'].values[0]
            
            scenario_base.append(base_rmse)
            scenario_cvap1.append(cvap1_rmse)
            scenario_cvap10.append(cvap10_rmse)
            
            # Statistical testing
            base_seeds = details_df[(details_df['scenario'] == scenario) & (details_df['model'] == reg)]['rmse'].values
            cvap1_seeds = details_df[(details_df['scenario'] == scenario) & (details_df['model'] == f"{reg} CVAP - 1")]['rmse'].values
            cvap10_seeds = details_df[(details_df['scenario'] == scenario) & (details_df['model'] == f"{reg} CVAP - 10")]['rmse'].values
            
            def get_stars(s1, s2):
                if len(s1) < 2 or len(s2) < 2: return ""
                t_stat, p_val = stats.ttest_rel(s1, s2)
                if p_val < 0.01: return "^{**}"
                if p_val < 0.05: return "^{*}"
                return ""

            star1 = get_stars(base_seeds, cvap1_seeds)
            star10 = get_stars(base_seeds, cvap10_seeds)
            
            # Bolding best
            vals = [base_rmse, cvap1_rmse, cvap10_rmse]
            best_val = min(vals)
            
            def fmt(val, std, is_best, stars=""):
                s = f"{val:.2f} \\pm {std:.2f}"
                if is_best and stars: s += stars
                if is_best: return f"$\\mathbf{{{s}}}$"
                return f"${s}$"

            row = f"{regressor_names[reg]} & {fmt(base_rmse, base_std, base_rmse == best_val)} & {fmt(cvap1_rmse, cvap1_std, cvap1_rmse == best_val, star1)} & {fmt(cvap10_rmse, cvap10_std, cvap10_rmse == best_val, star10)} \\\\"
            latex_rows.append(row)

        # Per-scenario average row
        s_avg_base = np.mean(scenario_base)
        s_avg_cvap1 = np.mean(scenario_cvap1)
        s_avg_cvap10 = np.mean(scenario_cvap10)
        
        _, p1 = stats.ttest_rel(scenario_base, scenario_cvap1)
        _, p10 = stats.ttest_rel(scenario_base, scenario_cvap10)
        
        def get_avg_stars(p):
            if p < 0.01: return "^{**}"
            if p < 0.05: return "^{*}"
            return ""

        star_avg1 = get_avg_stars(p1)
        star_avg10 = get_avg_stars(p10)
        
        best_avg = min([s_avg_base, s_avg_cvap1, s_avg_cvap10])
        
        def fmt_avg(val, is_best, stars=""):
            s = f"{val:.2f}"
            if is_best and stars: s += stars
            if is_best: return f"$\\mathbf{{{s}}}$"
            return f"${s}$"

        latex_rows.append(f"\\midrule\n\\textbf{{Average}} & {fmt_avg(s_avg_base, s_avg_base == best_avg)} & {fmt_avg(s_avg_cvap1, s_avg_cvap1 == best_avg, star_avg1)} & {fmt_avg(s_avg_cvap10, s_avg_cvap10 == best_avg, star_avg10)} \\\\")
    
    print("\n".join(latex_rows))

def generate_real_world_table():
    summary_df = pd.read_csv("output/real_datasets.csv")
    details_df = pd.read_csv("output/real_datasets_details.csv")
    
    datasets = summary_df['scenario'].unique()
    
    latex_rows = []
    
    regressors = ["LinearRegression", "Ridge", "Lasso", "ElasticNet", "RandomForest", "GradientBoosting", "SVR(RBF)"]
    regressor_names = {
        "LinearRegression": "Linear Regression", "Ridge": "Ridge", "Lasso": "Lasso",
        "ElasticNet": "ElasticNet", "RandomForest": "Random Forest",
        "GradientBoosting": "Gradient Boosting", "SVR(RBF)": "SVR (RBF)"
    }
    
    for ds in datasets:
        latex_rows.append(f"\\midrule\n\\multicolumn{{6}}{{c}}{{\\textbf{{Dataset: {ds.replace('_', ' ').title()}}}}} \\\\ \\midrule")
        
        for reg in regressors:
            base_row = summary_df[(summary_df['scenario'] == ds) & (summary_df['model'] == reg)]
            cvap1_row = summary_df[(summary_df['scenario'] == ds) & (summary_df['model'] == f"{reg} CVAP - 1")]
            cvap10_row = summary_df[(summary_df['scenario'] == ds) & (summary_df['model'] == f"{reg} CVAP - 10")]
            
            if base_row.empty: continue
            
            base_rmse = base_row['rmse_mean'].values[0]
            base_std = base_row['rmse_std'].values[0]
            base_calib = base_row['calib_err_mean'].values[0]
            base_calib_std = base_row['calib_err_std'].values[0]
            
            # Bolding best RMSE
            vals = [base_rmse]
            if not cvap1_row.empty: vals.append(cvap1_row['rmse_mean'].values[0])
            if not cvap10_row.empty: vals.append(cvap10_row['rmse_mean'].values[0])
            best_rmse = min(vals)
            
            def fmt(val, std, is_best, stars=""):
                s = f"{val:.2f} \\pm {std:.2f}"
                if is_best and stars: s += stars
                if is_best: return f"$\\mathbf{{{s}}}$"
                return f"${s}$"

            # Stars logic
            def get_stars_ds(ds_name, base_model, variant, metric='rmse'):
                s1 = details_df[(details_df['scenario'] == ds_name) & (details_df['model'] == base_model)][metric].values
                s2 = details_df[(details_df['scenario'] == ds_name) & (details_df['model'] == variant)][metric].values
                if len(s1) < 2 or len(s2) < 2: return ""
                _, p = stats.ttest_rel(s1, s2)
                if p < 0.01: return "^{**}"
                if p < 0.05: return "^{*}"
                return ""

            # Bolding best Calib Error
            c_vals = [base_calib]
            if not cvap1_row.empty: c_vals.append(cvap1_row['calib_err_mean'].values[0])
            if not cvap10_row.empty: c_vals.append(cvap10_row['calib_err_mean'].values[0])
            best_calib = min(c_vals)

            row_text = f"{regressor_names[reg]} & Base & {fmt(base_rmse, base_std, base_rmse == best_rmse)} & {fmt(base_calib, base_calib_std, base_calib == best_calib)} & --- & --- \\\\"
            latex_rows.append(row_text)
            
            if not cvap1_row.empty:
                c1_rmse = cvap1_row['rmse_mean'].values[0]
                c1_std = cvap1_row['rmse_std'].values[0]
                c1_calib = cvap1_row['calib_err_mean'].values[0]
                c1_calib_std = cvap1_row['calib_err_std'].values[0]
                c1_width = cvap1_row['width_mean_mean'].values[0]
                c1_width_std = cvap1_row['width_mean_std'].values[0]
                c1_med_width = cvap1_row['width_median_mean'].values[0]
                c1_med_width_std = cvap1_row['width_median_std'].values[0]
                star1 = get_stars_ds(ds, reg, f"{reg} CVAP - 1", 'rmse')
                star1_calib = get_stars_ds(ds, reg, f"{reg} CVAP - 1", 'calib_err')
                latex_rows.append(f"        & CVAP ($m=1$) & {fmt(c1_rmse, c1_std, c1_rmse == best_rmse, star1)} & {fmt(c1_calib, c1_calib_std, c1_calib == best_calib, star1_calib)} & ${c1_width:.2f} \\pm {c1_width_std:.2f}$ & ${c1_med_width:.2f} \\pm {c1_med_width_std:.2f}$ \\\\")
            
            if not cvap10_row.empty:
                c10_rmse = cvap10_row['rmse_mean'].values[0]
                c10_std = cvap10_row['rmse_std'].values[0]
                c10_calib = cvap10_row['calib_err_mean'].values[0]
                c10_calib_std = cvap10_row['calib_err_std'].values[0]
                c10_width = cvap10_row['width_mean_mean'].values[0]
                c10_width_std = cvap10_row['width_mean_std'].values[0]
                c10_med_width = cvap10_row['width_median_mean'].values[0]
                c10_med_width_std = cvap10_row['width_median_std'].values[0]
                star10 = get_stars_ds(ds, reg, f"{reg} CVAP - 10", 'rmse')
                star10_calib = get_stars_ds(ds, reg, f"{reg} CVAP - 10", 'calib_err')
                latex_rows.append(f"        & CVAP ($m=10$)& {fmt(c10_rmse, c10_std, c10_rmse == best_rmse, star10)} & {fmt(c10_calib, c10_calib_std, c10_calib == best_calib, star10_calib)} & ${c10_width:.2f} \\pm {c10_width_std:.2f}$ & ${c10_med_width:.2f} \\pm {c10_med_width_std:.2f}$ \\\\")
            
            latex_rows.append("\\midrule")

    print("\n".join(latex_rows))

if __name__ == "__main__":
    print("--- SYNTHETIC TABLE ---")
    generate_full_synthetic_table()
    print("\n--- REAL WORLD TABLE ---")
    generate_real_world_table()
