import time
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import copy

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skyburst import Job, Cluster, waiting_policy
from skyburst.node import Node

# ==========================================
# 1. Micro-Benchmark Setup
# ==========================================

def setup_mock_environment():
    """
    Creates a realistic dummy environment for benchmarking.
    """
    # 1. Create Cluster (100 Nodes)
    # Using the actual Cluster class to ensure we measure real iteration overhead
    num_nodes = 100
    cluster = Cluster(num_nodes=num_nodes, num_gpus_per_node=8, num_cpus_per_node=96)
    
    # Populate cluster state (Simulate 50% load)
    # This is important because K8s and Cardinal iterate over nodes
    for i in range(num_nodes):
        node = cluster.nodes[i]
        if i % 2 == 0:
            # Half nodes are full
            node.free_gpus = 0
            node.free_cpus = 0
        else:
            # Half nodes have some space
            node.free_gpus = 4
            node.free_cpus = 48
            
    # Mock active jobs for HEFT dependency check
    # We need to simulate that the parent job is running
    parent_job = Job(idx=999, resources={'GPUs': 4}, arrival=0, runtime=100)
    parent_job.start = 10
    cluster.active_jobs[999] = parent_job
            
    # 2. Create Dummy Job
    # GPU job with dependency
    job = Job(idx=1000, resources={'GPUs': 4, 'CPUs': 32}, arrival=50.0, runtime=300.0)
    job.dependency_parent = 999 # Points to active parent
    job.data_size_mb = 5000 # 5GB data
    job.deadline = 1000.0 # Relaxed deadline
    
    # Set attributes expected by Cardinal-Query
    job.gpu_util = 80.0
    job.mem_util = 40.0
    job.is_private = False
    
    return cluster, job

def run_benchmark():
    print("==========================================================")
    print("Starting Experiment 7: Scheduling Overhead Micro-benchmark")
    print("==========================================================")
    
    cluster, job = setup_mock_environment()
    
    # Ensure RL scheduler is configured (Training Mode = False)
    waiting_policy.configure_scheduler({'train_rl': False})
    
    # Define policies to test
    # We wrap them to unify signature
    policies = {
        'K8s-Native': lambda: waiting_policy.k8s_wait(job, cluster, cur_timestamp=50.0, waiting_factor=1.0, mode='native'),
        'HEFT': lambda: waiting_policy.heft_wait(job, cluster, cur_timestamp=50.0, waiting_factor=1.0),
        'IntentSky': lambda: waiting_policy.cardinal_query_wait(job, cluster, cur_timestamp=50.0, waiting_factor=1.0)
    }
    
    results = []
    
    ITERATIONS = 10000
    WARMUP = 100
    
    print(f"Configuration: {ITERATIONS} iterations, {WARMUP} warmup.")
    print(f"Cluster Size: {cluster.num_nodes} nodes.")
    
    for name, func in policies.items():
        print(f"Benchmarking {name}...")
        
        # Warmup
        for _ in range(WARMUP):
            func()
            
        # Measurement
        latencies_ns = []
        for _ in range(ITERATIONS):
            start = time.perf_counter_ns()
            func()
            end = time.perf_counter_ns()
            latencies_ns.append(end - start)
            
        # Convert to microseconds (us) for clearer visualization
        latencies_us = np.array(latencies_ns) / 1e3
        
        avg_lat = np.mean(latencies_us)
        std_lat = np.std(latencies_us)
        p99_lat = np.percentile(latencies_us, 99)
        
        results.append({
            'Policy': name,
            'Avg Latency (us)': avg_lat,
            'Std Dev (us)': std_lat,
            'P99 Latency (us)': p99_lat
        })
        
        print(f"  -> Avg: {avg_lat:.4f} us | P99: {p99_lat:.4f} us")

    # ==========================================
    # 2. Output Results
    # ==========================================
    df = pd.DataFrame(results)
    
    print("\n--- Scheduling Overhead Results ---")
    print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=".4f"))
    
    df.to_csv('overhead_results.csv', index=False)
    
    return df

# ==========================================
# 3. Visualization
# ==========================================

def plot_overhead(df):
    # Set academic plotting style
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("white")

    # 1. 全局学术字体与排版规范设置
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.2  # 加粗坐标轴线宽

    plt.figure(figsize=(6, 4))
    
    # Color palette (Morandi-like / Soft colors)
    # K8s: Soft Blue, HEFT: Soft Grey, IntentSky: Soft Red
    colors = ['#AEC7E8', '#C7C7C7', '#FF9896']
    hatches = ['///', '...', 'xx']
    
    # Create bar plot manually for fine-grained control
    bars = plt.bar(
        df['Policy'], 
        df['Avg Latency (us)'], 
        yerr=df['Std Dev (us)'],
        capsize=3,          # Smaller caps for cleaner look
        color=colors,
        alpha=0.9,
        edgecolor='black',
        linewidth=1.0,      # Thinner borders
        width=0.6           # Thinner bars for elegance
    )
    
    # Apply hatching patterns for texture
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    
    # Axis Labels
    plt.ylabel(r'Decision Latency ($\mu$s)', fontsize=14, fontweight='bold')
    plt.xlabel('') # Remove X-axis label as Policy names are self-explanatory
    
    # Despine: Remove Top and Right borders for minimalist look
    # sns.despine(top=True, right=True) # REMOVED to keep the box
    
    # Minimalist Annotation for IntentSky
    # Find the bar for IntentSky
    rl_row = df[df['Policy']=='IntentSky']
    if not rl_row.empty:
        rl_val = rl_row['Avg Latency (us)'].values[0]
        # Assuming IntentSky is the 3rd bar (index 2)
        rl_idx = 2 
        plt.text(rl_idx-0.02, rl_val + 2, # Small offset
                 f"~{int(rl_val)} $\mu$s", 
                 ha='center', va='bottom', fontsize=11, color='Green', fontweight='bold')

    # Grid: Horizontal only, faint, dashed
    plt.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
    
    # 强制全封闭边界框 (Bounding Box)
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    
    plt.tight_layout()
    
    plt.savefig('fig_overhead_analysis.pdf', bbox_inches='tight')
    print("Plot saved to fig_overhead_analysis.pdf")

if __name__ == "__main__":
    df = run_benchmark()
    plot_overhead(df)
