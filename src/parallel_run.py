import subprocess
import multiprocessing
import os
import sys
import time

def run_job(job_info):
    dataset_type, scenario, n_samples, noise_level, table_num = job_info
    
    cmd = [
        sys.executable, "src/main.py",
        "--dataset", dataset_type,
        "--n_seeds", "100",
        "--save_details",
        "--scenario", scenario
    ]
    
    if dataset_type == "synthetic_datasets":
        cmd += ["--n_samples", str(n_samples), "--noise_level", str(noise_level)]
        name = f"Table {table_num}: {scenario} (n={n_samples}, noise={noise_level})"
    else:
        # Real dataset
        name = f"Table {table_num}: {scenario} (Real)"
    
    # Each job will save its own CSVs in output/ via main.py logic
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        return True, name, None
    else:
        return False, name, result.stderr

if __name__ == "__main__":
    start_time = time.time()
    
    # 10 Core Synthetic Datasets (Table 1-40)
    core_synthetic = [
        'linear_gaussian', 'nonlinear_sine', 'heteroscedastic',
        'heavy_tailed', 'outliers', 'sparse_highdim', 'covariate_shift', 
        'friedman1', 'friedman2', 'friedman3'
    ]
    n_samples_list = [1000, 10000]
    noise_levels = [1, 3]
    
    all_tasks = []
    table_counter = 1
    
    # Tables 1-40: Synthetic Core
    for ds in core_synthetic:
        for samples in n_samples_list:
            for noise in noise_levels:
                all_tasks.append(["synthetic_datasets", ds, samples, noise, table_counter])
                table_counter += 1
                
    # Tables 41-44: Real World (As clarified)
    # 41: Electricity, 42: Bias Correction, 43: Wine Red (Replacement), 44: Airfoil
    all_tasks.append(["real_datasets", "electricity", None, None, 41])
    all_tasks.append(["real_datasets", "climate_bias", None, None, 42])
    all_tasks.append(["real_datasets", "wine_red", None, None, 43])
    all_tasks.append(["real_datasets", "airfoil", None, None, 44])
    
    # Tables 45-47: New Bounded Synthetic (Noise 1, 2, 3 at 10000 samples)
    all_tasks.append(["synthetic_datasets", "bounded_logistic", 10000, 1, 45])
    all_tasks.append(["synthetic_datasets", "bounded_logistic", 10000, 2, 46])
    all_tasks.append(["synthetic_datasets", "bounded_logistic", 10000, 3, 47])
    
    total_jobs = len(all_tasks)
    print(f"Starting experiment suite: {total_jobs} jobs total.")
    print(f"Using 7 cores. Estimated time: ~17.5 hours.")
    print("-" * 50)
    
    completed = 0
    errors = []
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    with multiprocessing.Pool(processes=7) as pool:
        # Use imap_unordered for immediate feedback as jobs finish
        for success, name, error_msg in pool.imap_unordered(run_job, all_tasks):
            completed += 1
            elapsed = time.time() - start_time
            avg_time = elapsed / completed
            remaining = (total_jobs - completed) * avg_time
            
            # Format time
            def fmt_time(seconds):
                h = int(seconds // 3600)
                m = int((seconds % 3600) // 60)
                s = int(seconds % 60)
                return f"{h:02d}:{m:02d}:{s:02d}"
            
            status = "SUCCESS" if success else "FAILED"
            print(f"[{fmt_time(elapsed)}] {status}: {name}")
            print(f"Progress: {completed}/{total_jobs} completed. ETA: {fmt_time(remaining)}")
            print("-" * 50)
            
            if not success:
                errors.append((name, error_msg))
    
    total_elapsed = time.time() - start_time
    print(f"\nFinal Report:")
    print(f"Total time: {fmt_time(total_elapsed)}")
    print(f"Successfully completed: {completed - len(errors)}/{total_jobs}")
    
    if errors:
        print(f"\nErrors encountered in {len(errors)} jobs:")
        for name, msg in errors:
            print(f"--- {name} ---\n{msg}\n")
        sys.exit(1)
    else:
        print("\nAll experiments completed successfully. Results saved in output/ directory.")
