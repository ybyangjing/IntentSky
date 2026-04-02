import pandas as pd
import os
import hashlib
import numpy as np
from tabulate import tabulate

def get_file_md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def analyze_distribution(df, col_name):
    if col_name not in df.columns:
        return "N/A"
    
    series = df[col_name]
    stats = {
        'Min': series.min(),
        'P25': series.quantile(0.25),
        'Median': series.median(),
        'Mean': series.mean(),
        'P75': series.quantile(0.75),
        'P99': series.quantile(0.99),
        'Max': series.max(),
        'Std': series.std()
    }
    return stats

def main():
    # Define file paths
    files = {
        "Root_Helios": r"e:\starburst_Relsing_Sky\integrated_helios_workload.csv",
        "Root_Philly": r"e:\starburst_Relsing_Sky\integrated_philly_workload.csv",
        "Data_Helios": r"e:\starburst_Relsing_Sky\data\integral\integrated_helios_workload.csv",
        "Data_Philly": r"e:\starburst_Relsing_Sky\data\integral\integrated_philly_workload.csv"
    }

    print("Step 1: File Existence and Integrity Check")
    print("-" * 60)
    
    file_info = []
    dfs = {}
    
    for name, path in files.items():
        if os.path.exists(path):
            md5 = get_file_md5(path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            file_info.append([name, "Exists", f"{size_mb:.2f} MB", md5])
            
            try:
                dfs[name] = pd.read_csv(path)
            except Exception as e:
                print(f"Error reading {name}: {e}")
        else:
            file_info.append([name, "MISSING", "N/A", "N/A"])

    print(tabulate(file_info, headers=["File Key", "Status", "Size", "MD5 Hash"], tablefmt="grid"))
    
    print("\nStep 2: Detailed Content Analysis (Duration & Resources)")
    print("-" * 60)
    
    # Analyze Duration
    duration_stats = []
    resource_stats = []
    
    for name, df in dfs.items():
        # Identify Duration Column
        dur_col = next((c for c in ['duration', 'runtime', 'run_time'] if c in df.columns), None)
        gpu_col = next((c for c in ['gpu_num', 'num_gpus', 'gpus'] if c in df.columns), None)
        
        if dur_col:
            stats = analyze_distribution(df, dur_col)
            row = [name, dur_col] + [f"{v:.2f}" for v in stats.values()]
            duration_stats.append(row)
            
        if gpu_col:
            stats = analyze_distribution(df, gpu_col)
            row = [name, gpu_col] + [f"{v:.2f}" for v in stats.values()]
            resource_stats.append(row)

    print("\n[Duration Statistics]")
    print(tabulate(duration_stats, headers=["Dataset", "Col", "Min", "P25", "Median", "Mean", "P75", "P99", "Max", "Std"], tablefmt="grid"))

    print("\n[Resource Statistics (GPU Num)]")
    print(tabulate(resource_stats, headers=["Dataset", "Col", "Min", "P25", "Median", "Mean", "P75", "P99", "Max", "Std"], tablefmt="grid"))

    # Check for Column differences
    print("\nStep 3: Column Difference Analysis")
    print("-" * 60)
    if "Root_Helios" in dfs and "Root_Philly" in dfs:
        cols_h = set(dfs["Root_Helios"].columns)
        cols_p = set(dfs["Root_Philly"].columns)
        
        print(f"Columns unique to Helios: {cols_h - cols_p}")
        print(f"Columns unique to Philly: {cols_p - cols_h}")
        print(f"Shared Columns: {cols_h.intersection(cols_p)}")

if __name__ == "__main__":
    main()
