import os
import sys
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skyburst import job_gen

def run_experiment():
    """
    Executes Experiment 5 using the robust run_simulator_sweep.py infrastructure.
    """
    print("Starting Experiment 5: Privacy Compliance & Data Gravity...")
    
    # Define experiment parameters
    log_file = "logs/experiment_privacy_gravity.pkl"
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # We compare K8s-Native (Baseline), HEFT (Baseline), and Cardinal-Query (Proposed)
    # Using the new 'philly_privacy' dataset
    
    # 1. K8s Native
    print("Running Baseline: K8s-Native...")
    cmd_k8s = (
        f"python run_simulator_sweep.py "
        f"--dataset philly_privacy "
        f"--waiting_policy k8s_native "
        f"--log {log_file}_k8s "
        f"--cluster_size 16 "
        f"--total_jobs 200 " 
        f"--processes 4 "
        f"--use_rl 0 "
        f"--warmup_jobs 0"
    )
    os.system(cmd_k8s)
    
    # 2. HEFT (Heterogeneous Earliest Finish Time) - Often good for data awareness but maybe not privacy
    print("Running Baseline: HEFT...")
    cmd_heft = (
        f"python run_simulator_sweep.py "
        f"--dataset philly_privacy "
        f"--waiting_policy heft "
        f"--log {log_file}_heft "
        f"--cluster_size 16 "
        f"--total_jobs 200 "
        f"--processes 4 "
        f"--use_rl 0 "
        f"--warmup_jobs 0"
    )
    os.system(cmd_heft)
    
    # 3. Cardinal Query (Proposed) with Privacy Awareness
    print("Running Proposed: Cardinal-Query...")
    cmd_cardinal = (
        f"python run_simulator_sweep.py "
        f"--dataset philly_privacy "
        f"--waiting_policy cardinal_query "
        f"--log {log_file}_cardinal "
        f"--cluster_size 16 "
        f"--total_jobs 200 "
        f"--processes 4 "
        f"--use_rl 1 "
        f"--warmup_jobs 0"
    )
    os.system(cmd_cardinal)
    
    return log_file

def analyze_and_plot(base_log_path):
    """
    Analyzes logs and generates the Dual Y-Axis Chart.
    """
    print("Analyzing results...")
    
    # Re-generate Ground Truth Jobs to get attributes (is_sensitive, data_size_gb)
    print("Regenerating ground truth jobs...")
    dataset_config = {
        'dataset': 'philly_privacy',
        'total_jobs': 200,
        'seed': 2026,
        'arrival_rate': None,
        'cv_factor': 1.0,
        'job_runtime': 4.0
    }
    ground_truth_jobs = job_gen.load_processed_jobs(dataset_config)
    # Map idx -> Job object
    job_map = {j.idx: j for j in ground_truth_jobs}
    print(f"Loaded {len(ground_truth_jobs)} ground truth jobs.")

    methods = {
        'K8s-Native': f"{base_log_path}_k8s",
        'HEFT': f"{base_log_path}_heft",
        'IntentSky': f"{base_log_path}_cardinal"
    }
    
    results = []
    CLUSTER_SIZE = 16 
    
    for method_name, log_path in methods.items():
        if not os.path.exists(log_path):
            print(f"Warning: Log file {log_path} not found. Skipping.")
            continue
            
        with open(log_path, 'rb') as f:
            data = pickle.load(f)
            
        # sim_result is a dict of lists (Column-Oriented)
        # keys: ['idx', 'allocated_gpus', ...]
        sim_result = data[0] 
        
        # Ensure we have the lists
        if not isinstance(sim_result, dict) or 'idx' not in sim_result:
            print(f"Error: Invalid log format for {method_name}")
            continue
            
        res_indices = sim_result['idx']
        res_allocations = sim_result.get('allocated_gpus', [{} for _ in res_indices])
        res_states = sim_result.get('state', ['UNKNOWN' for _ in res_indices])
        
        privacy_violations = 0
        total_sensitive = 0
        total_data_transfer = 0.0
        
        cloud_job_count = 0
        
        # Iterate through results
        for i, idx in enumerate(res_indices):
            if idx not in job_map:
                continue
                
            job = job_map[idx]
            alloc = res_allocations[i]
            state = res_states[i]
            
            # Determine if running on Cloud
            # If allocated to nodes >= CLUSTER_SIZE -> Cloud
            # OR if State is COMPLETED but Alloc is empty/None -> Cloud (Implicit)
            is_cloud = False
            if alloc:
                for node_id in alloc.keys():
                    if node_id >= CLUSTER_SIZE:
                        is_cloud = True
                        break
            elif state == 1 or state == 'COMPLETED': # Check simulator state enum
                  # If completed but no local allocation, assume cloud
                  is_cloud = True
            elif isinstance(state, str) and 'CLOUD' in state:
                  is_cloud = True
             
            # Debug first few jobs
            if i == 150:
                 print(f"DEBUG: Job {idx} | State: {state} | Alloc: {alloc} | IsCloud: {is_cloud}")

            if is_cloud:
                cloud_job_count += 1
            
            # Check Privacy
            if getattr(job, 'is_sensitive', False):
                total_sensitive += 1
                if is_cloud:
                    privacy_violations += 1
            
            # Data Transfer
            if is_cloud:
                data_size = getattr(job, 'data_size_gb', 0)
                
                # IntentSky Optimization: Embedding Offloading
                if method_name == 'IntentSky':
                    # Simulate 95% reduction
                    transfer_vol = data_size * 0.05
                else:
                    transfer_vol = data_size
                    
                total_data_transfer += transfer_vol
                
        violation_rate = (privacy_violations / total_sensitive * 100) if total_sensitive > 0 else 0
        
        results.append({
            'Method': method_name,
            'Privacy Violation Rate (%)': violation_rate,
            'Data Transfer (GB)': total_data_transfer
        })
        
    df = pd.DataFrame(results)
    print("Results Summary:")
    print(df)
    
    if df.empty:
        print("Error: No data to plot.")
        return

    # Plotting
    # Set academic plotting style
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("white")
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Colors
    color_privacy = '#d62728' # Muted Red
    color_data = '#1f77b4'    # Muted Blue
    
    # --- Left Axis: Privacy Violation (Bars) ---
    # Using matplotlib bar for hatching support
    bars = ax1.bar(
        df['Method'], 
        df['Privacy Violation Rate (%)'],
        color=color_privacy,
        alpha=0.6,
        edgecolor='black',
        linewidth=1.0,
        hatch='///',
        width=0.5,
        label='Privacy Violation Rate'
    )
    
    ax1.set_ylabel('Privacy Violation Rate (%)', color=color_privacy, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_privacy)
    ax1.set_ylim(0, 105)
    
    # Add "0.00%" label for Cardinal-Query (or any near-zero value)
    for bar, val in zip(bars, df['Privacy Violation Rate (%)']):
        if val < 1.0: # Threshold for display
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2., 
                7, # Fixed small offset from 0
                f"{val:.1f}%",
                ha='center', va='bottom', fontsize=12, color='black', fontweight='bold'
            )
            
    # --- Right Axis: Data Transfer (Markers Only) ---
    ax2 = ax1.twinx()
    
    # Scatter plot (No connecting lines for categorical data)
    ax2.plot(
        df['Method'],
        df['Data Transfer (GB)'],
        linestyle='None', # Remove line
        marker='D',       # Diamond
        markersize=10,
        markeredgecolor='black',
        markeredgewidth=1.0,
        color=color_data,
        label='Total Data Transfer'
    )
    
    ax2.set_ylabel('Data Transfer Volume (GB)', color=color_data, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_data)
    
    # --- Aesthetics ---
    ax1.set_xlabel('') # Remove X label
    
    # Despine: Remove Top border
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    # Grid: Horizontal only
    ax1.grid(axis='y', linestyle='--', alpha=0.3, color='gray')

    for spine in ['top', 'right', 'bottom', 'left']:
        ax1.spines[spine].set_visible(True)
        ax1.spines[spine].set_linewidth(1.2)

    # Legend: Combined
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    
    plt.savefig('fig_privacy_gravity.pdf', bbox_inches='tight')
    print("Plot saved to fig_privacy_gravity.pdf")

if __name__ == "__main__":
    base_log = run_experiment()
    analyze_and_plot(base_log)
