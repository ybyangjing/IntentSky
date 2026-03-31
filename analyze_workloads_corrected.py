import pandas as pd
import os
import sys
from tabulate import tabulate

def analyze_csv_detailed(file_path, name):
    print(f"Analyzing {name} from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)
    
    # Basic Stats
    stats = {
        "Dataset": name,
        "Total Jobs": len(df),
    }

    # 1. Duration Analysis
    # Handle varying column names
    dur_col = next((c for c in ['duration', 'runtime', 'run_time'] if c in df.columns), None)
    if dur_col:
        stats["Avg Duration (s)"] = f"{df[dur_col].mean():.2f}"
        stats["P50 Duration (s)"] = f"{df[dur_col].median():.2f}"
        stats["P99 Duration (s)"] = f"{df[dur_col].quantile(0.99):.2f}"
        stats["Max Duration (s)"] = f"{df[dur_col].max():.2f}"
        
        # Long vs Short Jobs
        long_threshold = 60 # 1 minute
        long_jobs = df[df[dur_col] > long_threshold]
        stats["Long Jobs (>60s)"] = f"{len(long_jobs)} ({len(long_jobs)/len(df)*100:.1f}%)"
    else:
        stats["Avg Duration (s)"] = "N/A"

    # 2. Resource Analysis (GPU/CPU)
    gpu_col = next((c for c in ['gpu_num', 'num_gpus', 'gpus'] if c in df.columns), None)
    if gpu_col:
        gpu_counts = df[gpu_col].value_counts().sort_index()
        # 0-GPU jobs are CPU-only
        cpu_only = gpu_counts.get(0, 0)
        gpu_jobs = len(df) - cpu_only
        
        stats["Total GPU Jobs"] = f"{gpu_jobs} ({gpu_jobs/len(df)*100:.1f}%)"
        stats["Total CPU-Only Jobs"] = f"{cpu_only} ({cpu_only/len(df)*100:.1f}%)"
        stats["Avg GPUs/Job"] = f"{df[gpu_col].mean():.2f}"
    else:
        stats["Total GPU Jobs"] = "N/A"

    # 3. Task Structure / Dependencies
    # Check for DAG structure
    if 'dependency_parent' in df.columns:
        # Count non-empty/non-null dependencies
        # Assuming format is string "123,124" or list
        # Or if it's NaN for no dependency
        has_dep = df['dependency_parent'].notna() & (df['dependency_parent'] != '') & (df['dependency_parent'] != '[]')
        dep_count = has_dep.sum()
        stats["DAG/Dependent Jobs"] = f"{dep_count} ({dep_count/len(df)*100:.1f}%)"
        
        # Check Task Roles
        if 'task_role' in df.columns:
            roles = df['task_role'].value_counts().head(3).to_dict()
            stats["Top Task Roles"] = str(roles)
    else:
        stats["DAG/Dependent Jobs"] = "0 (Flat)"

    # 4. Time Span
    time_col = next((c for c in ['submit_time', 'submitted_time', 'arrival_time'] if c in df.columns), None)
    if time_col:
        # Try numeric conversion
        try:
            times = pd.to_numeric(df[time_col], errors='coerce')
            if times.notna().sum() > 0:
                span = times.max() - times.min()
                stats["Time Span (Hours)"] = f"{span/3600:.2f}"
            else:
                # Try datetime
                times = pd.to_datetime(df[time_col], errors='coerce')
                span = (times.max() - times.min()).total_seconds()
                stats["Time Span (Hours)"] = f"{span/3600:.2f}"
        except:
            stats["Time Span (Hours)"] = "N/A"

    return stats

def main():
    # Use the Root files as verified in Step 1
    files = [
        (r"e:\starburst_Relsing_Sky\integrated_helios_workload.csv", "Helios (Root)"),
        (r"e:\starburst_Relsing_Sky\integrated_philly_workload.csv", "Philly (Root)")
    ]

    results = []
    for path, name in files:
        res = analyze_csv_detailed(path, name)
        if res:
            results.append(res)

    if not results:
        return

    # Pivot Table
    all_keys = []
    for r in results:
        for k in r.keys():
            if k not in all_keys and k != "Dataset":
                all_keys.append(k)
    
    # Custom Sort Order
    priority = [
        "Total Jobs", "Time Span (Hours)", 
        "Avg Duration (s)", "P50 Duration (s)", "P99 Duration (s)", "Max Duration (s)", "Long Jobs (>60s)",
        "Total GPU Jobs", "Total CPU-Only Jobs", "Avg GPUs/Job",
        "DAG/Dependent Jobs", "Top Task Roles"
    ]
    sorted_keys = [k for k in priority if k in all_keys] + [k for k in all_keys if k not in priority]

    table_data = []
    for k in sorted_keys:
        row = [k]
        for res in results:
            row.append(res.get(k, "-"))
        table_data.append(row)

    print("\n" + "="*60)
    print("Corrected Workload Analysis (Root Files)")
    print("="*60)
    print(tabulate(table_data, headers=["Metric"] + [r['Dataset'] for r in results], tablefmt="grid"))

if __name__ == "__main__":
    main()
