import pandas as pd
import numpy as np
import json
import os
from scipy.stats import ttest_rel

# Mapping Paper Table Num -> (scenario, n, noise)
PAPER_MAPPING = {}
scenarios = [
    'linear_gaussian', 'nonlinear_sine', 'heteroscedastic',
    'heavy_tailed', 'outliers', 'sparse_highdim', 'covariate_shift', 
    'friedman1', 'friedman2', 'friedman3'
]

# 1-10: 10k, 3
for i, sc in enumerate(scenarios):
    PAPER_MAPPING[i+1] = (sc, 10000, 3)
# 11-20: 10k, 1
for i, sc in enumerate(scenarios):
    PAPER_MAPPING[i+11] = (sc, 10000, 1)
# 21-30: 1k, 3
for i, sc in enumerate(scenarios):
    PAPER_MAPPING[i+21] = (sc, 1000, 3)
# 31-40: 1k, 1
for i, sc in enumerate(scenarios):
    PAPER_MAPPING[i+31] = (sc, 1000, 1)

MODEL_MAP = {
    "ElasticNet": "Elastic Net",
    "GradientBoosting": "Gradient Boosting",
    "Lasso": "Lasso",
    "LinearRegression": "Linear Regression",
    "RandomForest": "Random Forest",
    "Ridge": "Ridge",
    "SVR (RBF)": "SVR (RBF)",
    "average": "average"
}

def get_tex_table(df_mean, df_std=None, p_values=None, title=""):
    """
    df_mean: index=Model, cols=['Base', 'CVAR1', 'CVAR10']
    p_values: index=Model, cols=['CVAR1', 'CVAR10']
    """
    tex = "\\footnotesize\n"
    tex += "\\begin{tabular}{lccc}\n"
    tex += "\\toprule\n"
    tex += f"\\multicolumn{{4}}{{c}}{{{title}}} \\\\\n"
    tex += "\\midrule\n"
    tex += "Model & none & CVAR1 & CVAR10 \\\\\n"
    tex += "\\midrule\n"
    
    for model in df_mean.index:
        if model == "average":
            tex += "\\midrule\n"
            
        row_vals = df_mean.loc[model].values
        # Only bold if it's the best in the row
        min_idx = np.argmin(row_vals)
        
        row_str = f"{model}"
        for i, val in enumerate(row_vals):
            col_name = df_mean.columns[i]
            
            # SEM calculation if std provided
            if df_std is not None:
                sem = df_std.loc[model].values[i] / 10.0 # sqrt(100)
                cell = f"{val:.3f} \\pm {sem:.3f}"
                
                # Significance markers for Table C
                if p_values is not None and i > 0:
                    if model in p_values.index and col_name in p_values.columns:
                        p = p_values.loc[model, col_name]
                        if val < row_vals[0]: # Only if better than base
                            if p < 0.01:
                                cell += "^{**}"
                            elif p < 0.05:
                                cell += "^{*}"
                cell = f"${cell}$"
            else:
                cell = f"{val:.3f}"
            
            if i == min_idx:
                if cell.startswith("$"):
                    row_str += f" & \\bm{{{cell}}}"
                else:
                    row_str += f" & \\textbf{{{cell}}}"
            else:
                row_str += f" & {cell}"
        row_str += " \\\\\n"
        tex += row_str
        
    tex += "\\bottomrule\n"
    tex += "\\end{tabular}\n"
    return tex

def generate_rebuttal():
    with open("scratch/reference_tables.json", "r") as f:
        ref_data = json.load(f)
    
    # 3. Friedman Swap (Reference ONLY)
    # Pairs (9/10, 19/20, 29/30, 39/40)
    for p1, p2 in [(9, 10), (19, 20), (29, 30), (39, 40)]:
        s1, s2 = str(p1), str(p2)
        if s1 in ref_data and s2 in ref_data:
            ref_data[s1], ref_data[s2] = ref_data[s2], ref_data[s1]

    latex_output = "\\documentclass[a4paper,landscape]{article}\n"
    latex_output += "\\usepackage[utf8]{inputenc}\n"
    latex_output += "\\usepackage{booktabs}\n"
    latex_output += "\\usepackage{geometry}\n"
    latex_output += "\\geometry{margin=0.4in}\n"
    latex_output += "\\usepackage{subcaption}\n"
    latex_output += "\\setlength{\\tabcolsep}{4pt}\n"
    latex_output += "\\usepackage{bm}\n"
    latex_output += "\\begin{document}\n"
    latex_output += "\\title{IVAR Rebuttal Comparison Tables (100 Seeds)}\n"
    latex_output += "\\maketitle\n"
    latex_output += "\\newpage\n"

    for table_num in range(1, 41):
        sc, n, noise = PAPER_MAPPING[table_num]
        
        # 1. Reference Table (Table A)
        ref_table = ref_data.get(str(table_num), {})
        if not ref_table: continue
        
        ref_df = pd.DataFrame.from_dict(ref_table, orient='index', columns=['Base', 'CVAR1', 'CVAR10'])
        model_order = ["Elastic Net", "Gradient Boosting", "Lasso", "Linear Regression", "Random Forest", "Ridge", "SVR (RBF)", "average"]
        ref_df = ref_df.reindex(model_order).dropna()
        
        # 2. Experimental Data (Table B & C)
        csv_path = f"output/synthetic_datasets_{sc}_noise_{noise}_{n}_details.csv"
        if not os.path.exists(csv_path): continue
            
        df_details = pd.read_csv(csv_path)
        
        # Pivot to get raw values for t-test
        # df_details has cols: scenario, seed, model, rmse, ...
        # We need model names mapped to columns
        pivoted = df_details.pivot(index='seed', columns='model', values='rmse')
        
        final_mean = pd.DataFrame(index=model_order[:-1], columns=['Base', 'CVAR1', 'CVAR10'])
        final_std = pd.DataFrame(index=model_order[:-1], columns=['Base', 'CVAR1', 'CVAR10'])
        p_values = pd.DataFrame(index=model_order[:-1], columns=['CVAR1', 'CVAR10'])
        
        for m_orig in model_order[:-1]:
            m_our = m_orig.replace(" ", "")
            if m_our in pivoted.columns:
                base_vals = pivoted[m_our]
                final_mean.loc[m_orig, 'Base'] = base_vals.mean()
                final_std.loc[m_orig, 'Base'] = base_vals.std()
                
                c1_name = f"{m_our} CVAP - 1"
                if c1_name in pivoted.columns:
                    c1_vals = pivoted[c1_name]
                    final_mean.loc[m_orig, 'CVAR1'] = c1_vals.mean()
                    final_std.loc[m_orig, 'CVAR1'] = c1_vals.std()
                    # Paired T-test
                    _, p = ttest_rel(base_vals, c1_vals)
                    p_values.loc[m_orig, 'CVAR1'] = p
                
                c10_name = f"{m_our} CVAP - 10"
                if c10_name in pivoted.columns:
                    c10_vals = pivoted[c10_name]
                    final_mean.loc[m_orig, 'CVAR10'] = c10_vals.mean()
                    final_std.loc[m_orig, 'CVAR10'] = c10_vals.std()
                    _, p = ttest_rel(base_vals, c10_vals)
                    p_values.loc[m_orig, 'CVAR10'] = p
        
        # Add average row
        final_mean.loc['average'] = final_mean.mean()
        final_std.loc['average'] = final_std.mean()
        
        # Calculate t-test for average row
        # Average RMSE per seed across all models
        base_models = [m.replace(" ", "") for m in model_order[:-1]]
        avg_seed_base = pivoted[base_models].mean(axis=1)
        
        for col_our, col_paper in [('CVAP - 1', 'CVAR1'), ('CVAP - 10', 'CVAR10')]:
            c_models = [f"{m} {col_our}" for m in base_models]
            # filter only existing columns
            c_models = [m for m in c_models if m in pivoted.columns]
            if c_models:
                avg_seed_c = pivoted[c_models].mean(axis=1)
                _, p = ttest_rel(avg_seed_base, avg_seed_c)
                p_values.loc['average', col_paper] = p
        
        # Generate LaTeX
        latex_output += f"\\section*{{Table {table_num}: {sc.replace('_', '\\_')} (n={n}, $\\sigma={noise}$)}}\n"
        latex_output += "\\begin{figure}[h]\n"
        latex_output += "  \\centering\n"
        
        for df, std, p, cap, title in [
            (ref_df, None, None, "Paper Reference (Swapped F2/F3)", "Original (10 Seeds)"),
            (final_mean, None, None, "Our Results (Mean)", "Our Mean (100 Seeds)"),
            (final_mean, final_std, p_values, "Our Results (Mean $\\pm$ SEM + Sig)", "Our Mean $\\pm$ SEM")
        ]:
            latex_output += "  \\begin{subtable}[b]{0.32\\textwidth}\n"
            latex_output += "    \\centering\n"
            latex_output += get_tex_table(df, std, p, title=title)
            latex_output += f"    \\caption{{{cap}}}\n"
            latex_output += "  \\end{subtable}\n"
            if cap != "Our Results (Mean $\\pm$ SEM + Sig)":
                latex_output += "  \\hfill\n"
        
        latex_output += "\\end{figure}\n"
        latex_output += "\\vspace{0.5cm}\n"
        if table_num % 2 == 0: latex_output += "\\newpage\n"

    # --- Section: Real World Datasets ---
    latex_output += "\\newpage\\section*{Real-World Benchmarks}\n"
    
    # Mapping for Real World
    real_tasks = [
        (42, "climate_bias", "climate_bias", "Bias Correction (UCI)"),
        (44, "airfoil", "airfoil", "Airfoil Self-Noise (UCI)"),
        (43, "star", "star", "Student Performance (STAR)"),
        (None, "wine_red", "wine_red", "Wine Quality (Red)")
    ]
    
    for table_num, sc, csv_sc, title_name in real_tasks:
        csv_path = f"output/real_datasets_{csv_sc}_details.csv"
        # Fallback for old climate bias file name if needed
        if csv_sc == "climate_bias" and not os.path.exists(csv_path):
            csv_path = "output/real_datasets_details.csv"
            
        if not os.path.exists(csv_path): continue
        
        df_details = pd.read_csv(csv_path)
        pivoted = df_details.pivot(index='seed', columns='model', values='rmse')
        
        final_mean = pd.DataFrame(index=model_order[:-1], columns=['Base', 'CVAR1', 'CVAR10'])
        final_std = pd.DataFrame(index=model_order[:-1], columns=['Base', 'CVAR1', 'CVAR10'])
        p_values = pd.DataFrame(index=model_order[:-1], columns=['CVAR1', 'CVAR10'])
        
        for m_orig in model_order[:-1]:
            m_our = m_orig.replace(" ", "")
            if m_our in pivoted.columns:
                base_vals = pivoted[m_our]
                final_mean.loc[m_orig, 'Base'] = base_vals.mean()
                final_std.loc[m_orig, 'Base'] = base_vals.std()
                for c_paper, c_our in [('CVAR1', 'CVAP - 1'), ('CVAR10', 'CVAP - 10')]:
                    c_name = f"{m_our} {c_our}"
                    if c_name in pivoted.columns:
                        c_vals = pivoted[c_name]
                        final_mean.loc[m_orig, c_paper] = c_vals.mean()
                        final_std.loc[m_orig, c_paper] = c_vals.std()
                        _, p = ttest_rel(base_vals, c_vals)
                        p_values.loc[m_orig, c_paper] = p
        
        final_mean.loc['average'] = final_mean.mean()
        final_std.loc['average'] = final_std.mean()
        # Avg significance
        base_models = [m.replace(" ", "") for m in model_order[:-1]]
        avg_seed_base = pivoted[base_models].mean(axis=1)
        for c_paper, c_our in [('CVAR1', 'CVAP - 1'), ('CVAR10', 'CVAP - 10')]:
            c_models = [f"{m} {c_our}" for m in base_models if f"{m} {c_our}" in pivoted.columns]
            if c_models:
                avg_seed_c = pivoted[c_models].mean(axis=1)
                _, p = ttest_rel(avg_seed_base, avg_seed_c)
                p_values.loc['average', c_paper] = p

        if sc == "star":
            latex_output += "\\newpage\n"
        latex_output += f"\\subsection*{{{title_name}}}\n"
        latex_output += "\\begin{figure}[h]\n  \\centering\n"
        
        has_ref = table_num is not None and str(table_num) in ref_data
        if has_ref:
            ref_table = ref_data[str(table_num)]
            ref_df = pd.DataFrame.from_dict(ref_table, orient='index', columns=['Base', 'CVAR1', 'CVAR10']).reindex(model_order).dropna()
            
            # Subtable A
            latex_output += "  \\begin{subtable}[b]{0.32\\textwidth}\n    \\centering\n"
            latex_output += get_tex_table(ref_df, title="Original Paper")
            latex_output += f"    \\caption{{Paper Reference (Table {table_num})}}\n  \\end{{subtable}}\\hfill\n"
            
        # Subtable B & C
        sub_width = "0.32" if has_ref else "0.48"
        latex_output += f"  \\begin{{subtable}}[b]{{{sub_width}\\textwidth}}\n    \\centering\n"
        latex_output += get_tex_table(final_mean, title="Our Mean")
        latex_output += f"    \\caption{{Our Results (Mean)}}\n  \\end{{subtable}}\\hfill\n"
        
        latex_output += f"  \\begin{{subtable}}[b]{{{sub_width}\\textwidth}}\n    \\centering\n"
        latex_output += get_tex_table(final_mean, final_std, p_values, title="Our Mean $\\pm$ SEM")
        latex_output += f"    \\caption{{Our Results (Mean $\\pm$ SEM)}}\n  \\end{{subtable}}\n"
        latex_output += "\\end{figure}\n\\vspace{0.5cm}\n"

    # --- Section: New Bounded Synthetic ---
    latex_output += "\\newpage\\section*{New Bounded Synthetic Experiments (Tables 45--49)}\n"
    
    bounded_tasks = [
        (45, "bounded_logistic", 10000, 1),
        (47, "bounded_logistic", 10000, 3),
        (48, "bounded_logistic", 1000, 1),
        (49, "bounded_logistic", 1000, 3),
    ]
    
    for table_num, sc, n, noise in bounded_tasks:
        csv_path = f"output/synthetic_datasets_{sc}_noise_{noise}_{n}_details.csv"
        if not os.path.exists(csv_path): continue
            
        df_details = pd.read_csv(csv_path)
        pivoted = df_details.pivot(index='seed', columns='model', values='rmse')
        
        final_mean = pd.DataFrame(index=model_order[:-1], columns=['Base', 'CVAR1', 'CVAR10'])
        final_std = pd.DataFrame(index=model_order[:-1], columns=['Base', 'CVAR1', 'CVAR10'])
        p_values = pd.DataFrame(index=model_order[:-1], columns=['CVAR1', 'CVAR10'])
        
        for m_orig in model_order[:-1]:
            m_our = m_orig.replace(" ", "")
            if m_our in pivoted.columns:
                base_vals = pivoted[m_our]
                final_mean.loc[m_orig, 'Base'] = base_vals.mean()
                final_std.loc[m_orig, 'Base'] = base_vals.std()
                for c_paper, c_our in [('CVAR1', 'CVAP - 1'), ('CVAR10', 'CVAP - 10')]:
                    c_name = f"{m_our} {c_our}"
                    if c_name in pivoted.columns:
                        c_vals = pivoted[c_name]
                        final_mean.loc[m_orig, c_paper] = c_vals.mean()
                        final_std.loc[m_orig, c_paper] = c_vals.std()
                        _, p = ttest_rel(base_vals, c_vals)
                        p_values.loc[m_orig, c_paper] = p
        
        final_mean.loc['average'] = final_mean.mean()
        final_std.loc['average'] = final_std.mean()
        # Avg significance
        base_models = [m.replace(" ", "") for m in model_order[:-1]]
        avg_seed_base = pivoted[base_models].mean(axis=1)
        for c_paper, c_our in [('CVAR1', 'CVAP - 1'), ('CVAR10', 'CVAP - 10')]:
            c_models = [f"{m} {c_our}" for m in base_models if f"{m} {c_our}" in pivoted.columns]
            if c_models:
                avg_seed_c = pivoted[c_models].mean(axis=1)
                _, p = ttest_rel(avg_seed_base, avg_seed_c)
                p_values.loc['average', c_paper] = p

        latex_output += f"\\subsection*{{Table {table_num}: {sc.replace('_', '\\_')} (n={n}, noise={noise})}}\n"
        latex_output += "\\begin{figure}[h]\n  \\centering\n"
        
        # 2 tables only
        latex_output += "  \\begin{subtable}[b]{0.48\\textwidth}\n    \\centering\n"
        latex_output += get_tex_table(final_mean, title="Our Mean")
        latex_output += f"    \\caption{{Our Results (Mean)}}\n  \\end{{subtable}}\\hfill\n"
        
        latex_output += "  \\begin{subtable}[b]{0.48\\textwidth}\n    \\centering\n"
        latex_output += get_tex_table(final_mean, final_std, p_values, title="Our Mean $\\pm$ SEM")
        latex_output += f"    \\caption{{Our Results (Mean $\\pm$ SEM)}}\n  \\end{{subtable}}\n"
        latex_output += "\\end{figure}\n\\vspace{0.5cm}\n"
        if table_num % 2 == 1: latex_output += "\\newpage\n"

    latex_output += "\\end{document}\n"
    
    with open("output/generate_tables.tex", "w") as f:
        f.write(latex_output)
    print("Generated output/generate_tables.tex")

if __name__ == "__main__":
    generate_rebuttal()
