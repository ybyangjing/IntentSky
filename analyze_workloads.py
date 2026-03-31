import pandas as pd
import os
import sys
from tabulate import tabulate

def analyze_csv(file_path, name):
    print(f"Analyzing {name} from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    stats = {
        "Dataset": name,
        "Total Jobs": len(df),
        "Columns": list(df.columns)
    }

    # Time span (assuming 'submit_time' or similar exists)
    # Common columns in these traces: 'submit_time', 'duration', 'gpu_num', 'cpu_num' (or similar)
    # Let's inspect columns dynamically.
    
    # 1. Job Arrival / Submission
    time_col = None
    for col in ['submit_time', 'arrival_time', 'timestamp', 'submitted_time']:
        if col in df.columns:
            time_col = col
            break
            
    if time_col:
        # Convert to datetime if it's object/string type
        if df[time_col].dtype == 'object':
            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except Exception as e:
                print(f"Warning: Could not parse datetime for {time_col}: {e}")
                stats["Time Span (Hours)"] = "N/A (Parse Error)"
                time_col = None

    if time_col:
        min_time = df[time_col].min()
        max_time = df[time_col].max()
        
        # If datetime objects
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            duration_hours = (max_time - min_time).total_seconds() / 3600
        else:
            # Assume numeric (seconds or similar)
            duration_hours = (max_time - min_time) / 3600 if max_time > min_time else 0
            
        stats["Time Span (Hours)"] = f"{duration_hours:.2f}"
    else:
        stats["Time Span (Hours)"] = "N/A"

    # 2. Job Duration / Runtime
    duration_col = None
    for col in ['duration', 'runtime', 'run_time', 'execution_time']:
        if col in df.columns:
            duration_col = col
            break
            
    if duration_col:
        stats["Avg Duration (s)"] = f"{df[duration_col].mean():.2f}"
        stats["Max Duration (s)"] = f"{df[duration_col].max():.2f}"
        stats["P99 Duration (s)"] = f"{df[duration_col].quantile(0.99):.2f}"
    else:
        stats["Avg Duration (s)"] = "N/A"

    # 3. Resources (GPU)
    gpu_col = None
    for col in ['gpu_num', 'num_gpus', 'gpu_count', 'gpus']:
        if col in df.columns:
            gpu_col = col
            break
            
    if gpu_col:
        stats["Total GPU Hours"] = f"{(df[gpu_col] * df[duration_col]).sum() / 3600:.2f}" if duration_col else "N/A"
        stats["Avg GPUs per Job"] = f"{df[gpu_col].mean():.2f}"
        stats["Max GPUs per Job"] = f"{df[gpu_col].max()}"
        stats["Jobs using GPUs"] = f"{df[df[gpu_col] > 0].shape[0]} ({df[df[gpu_col] > 0].shape[0]/len(df)*100:.1f}%)"
    else:
        stats["Avg GPUs per Job"] = "N/A"

    # 4. Resources (CPU) - optional if focused on GPU
    cpu_col = None
    for col in ['cpu_num', 'num_cpus', 'cpu_count', 'cpus']:
        if col in df.columns:
            cpu_col = col
            break
    
    if cpu_col:
        stats["Avg CPUs per Job"] = f"{df[cpu_col].mean():.2f}"
    else:
        stats["Avg CPUs per Job"] = "N/A"
        
    # 5. Status (if available)
    status_col = None
    for col in ['status', 'state', 'job_status']:
        if col in df.columns:
            status_col = col
            break
    
    if status_col:
        # Just top statuses
        top_status = df[status_col].value_counts().head(3).to_dict()
        stats["Top Statuses"] = str(top_status)

    # 6. Specific Meta Columns
    if 'meta_is_multimodal' in df.columns:
        multimodal_count = df['meta_is_multimodal'].sum()
        stats["Multimodal Jobs"] = f"{multimodal_count} ({multimodal_count/len(df)*100:.1f}%)"
    
    if 'gpu_util_avg' in df.columns:
        stats["Avg GPU Util (%)"] = f"{df['gpu_util_avg'].mean():.2f}"
        
    if 'mem_util_avg' in df.columns:
        stats["Avg Mem Util (%)"] = f"{df['mem_util_avg'].mean():.2f}"

    return stats

def main():
    files = [
        (r"e:\starburst_Relsing_Sky\data\integral\integrated_helios_workload.csv", "Helios"),
        (r"e:\starburst_Relsing_Sky\data\integral\integrated_philly_workload.csv", "Philly")
    ]

    results = []
    for path, name in files:
        res = analyze_csv(path, name)
        if res:
            results.append(res)

    if not results:
        print("No results to display.")
        return

    # Pivot for better display (Metrics as rows, Datasets as columns)
    # Union of all keys from all results
    all_keys = set()
    for res in results:
        all_keys.update(res.keys())
    
    # Remove 'Dataset' and 'Columns'
    display_keys = [k for k in sorted(list(all_keys)) if k not in ['Dataset', 'Columns']]
    
    # Sort keys for logical grouping (optional, but nice)
    # Let's just use the sorted order for now, or manually order common ones
    priority_order = ["Total Jobs", "Time Span (Hours)", "Avg Duration (s)", "P99 Duration (s)", "Multimodal Jobs", "Avg GPU Util (%)", "Avg Mem Util (%)"]
    sorted_keys = []
    for k in priority_order:
        if k in display_keys:
            sorted_keys.append(k)
            display_keys.remove(k)
    sorted_keys.extend(display_keys)
    
    table_data = []
    for k in sorted_keys:
        row = [k]
        for res in results:
            row.append(res.get(k, "N/A"))
        table_data.append(row)

    print("\n" + "="*50)
    print("Workload Statistics Summary")
    print("="*50)
    print(tabulate(table_data, headers=["Metric"] + [r['Dataset'] for r in results], tablefmt="grid"))
    
    print("\n" + "="*50)
    print("Column Schema Info")
    print("="*50)
    for res in results:
        print(f"\nDataset: {res['Dataset']}")
        print(f"Columns: {', '.join(res['Columns'])}")

if __name__ == "__main__":
    main()
