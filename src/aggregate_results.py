import pandas as pd
import glob
import os

def aggregate_results(dataset_group, noise_level, n_samples):
    pattern = f"output/{dataset_group}_*_noise_{noise_level}_{n_samples}.csv"
    files = glob.glob(pattern)
    # Filter out files that have '_details' in them
    summary_files = [f for f in files if "_details" not in f]
    
    dfs = []
    for f in summary_files:
        df = pd.read_csv(f, index_col=0) if 'Unnamed: 0' in pd.read_csv(f, nrows=1).columns else pd.read_csv(f)
        dfs.append(df)
    
    if not dfs:
        print(f"No summary files found for {pattern}")
        return
    
    full_summary = pd.concat(dfs, ignore_index=True)
    # Sort by scenario and model
    full_summary = full_summary.sort_values(['scenario', 'model'])
    
    output_path = f"output/{dataset_group}_noise_{noise_level}_{n_samples}.csv"
    full_summary.to_csv(output_path, index=False)
    print(f"Aggregated summary saved to {output_path}")

    # Aggregating details files
    pattern_details = f"output/{dataset_group}_*_noise_{noise_level}_{n_samples}_details.csv"
    details_files = glob.glob(pattern_details)
    dfs_details = []
    for f in details_files:
        dfs_details.append(pd.read_csv(f))
    
    if dfs_details:
        full_details = pd.concat(dfs_details, ignore_index=True)
        output_details_path = f"output/{dataset_group}_noise_{noise_level}_{n_samples}_details.csv"
        full_details.to_csv(output_details_path, index=False)
        print(f"Aggregated details saved to {output_details_path}")

if __name__ == "__main__":
    aggregate_results("synthetic_datasets", 1, 10000)
