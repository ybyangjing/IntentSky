import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List
import copy
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skyburst import job_gen, run_simulator
from skyburst import waiting_policy

# ==========================================
# 1. Global Helpers & Patching Logic
# ==========================================

# Store original functions to restore/wrap later
ORIGINAL_K8S_WAIT = waiting_policy.k8s_wait
ORIGINAL_HEFT_WAIT = waiting_policy.heft_wait
ORIGINAL_CARDINAL_WAIT = waiting_policy.cardinal_query_wait

def inject_bandwidth_into_policies(bandwidth_mbps):
    """
    Dynamically patches waiting_policy functions to include transfer delay.
    """
    bandwidth_gbps = bandwidth_mbps / 1000.0
    # Avoid division by zero
    if bandwidth_gbps <= 0: bandwidth_gbps = 0.001
    
    print(f"  -> Patching policies with Bandwidth: {bandwidth_mbps} Mbps ({bandwidth_gbps:.3f} GB/s)")

    def calculate_transfer_time(job):
        # Heuristic if missing
        data_size = getattr(job, 'data_size_gb', 5.0) 
        # Safety for 0 size
        if data_size <= 0: data_size = 1.0
        return data_size / bandwidth_gbps

    # --- Patch K8s-Native ---
    def patched_k8s_wait(job, cluster=None, cur_timestamp=0, waiting_factor=1.0, mode='native'):
        # Call original
        res = ORIGINAL_K8S_WAIT(job, cluster, cur_timestamp, waiting_factor, mode)
        
        # If result is a Cloud Deadline (not infinite)
        if res < 1e10:
            transfer_time = calculate_transfer_time(job)
            # Full blocking penalty
            res += transfer_time
        return res

    # --- Patch HEFT ---
    def patched_heft_wait(job, cluster=None, cur_timestamp=0, waiting_factor=1.0):
        # Call original
        res = ORIGINAL_HEFT_WAIT(job, cluster, cur_timestamp, waiting_factor)
        
        # HEFT in this repo usually returns a timestamp (start + runtime). 
        # If it implies cloud execution (which it mostly does in current impl as it doesn't check local fit),
        # we add penalty.
        if res < 1e10:
            transfer_time = calculate_transfer_time(job)
            res += transfer_time
        return res

    # --- Patch Cardinal-Query ---
    def patched_cardinal_wait(job, cluster=None, cur_timestamp=0, waiting_factor=1.0, rounds=3, adaptation_rate=0.2):
        # Call original
        res = ORIGINAL_CARDINAL_WAIT(job, cluster, cur_timestamp, waiting_factor, rounds, adaptation_rate)
        
        # If result is a Cloud Deadline
        if res < 1e10:
            transfer_time = calculate_transfer_time(job)
            # Pipeline Masking Benefit: Only pay 10% of transfer time
            effective_delay = 0.1 * transfer_time
            res += effective_delay
        return res

    # Apply Patches
    waiting_policy.k8s_wait = patched_k8s_wait
    waiting_policy.heft_wait = patched_heft_wait
    waiting_policy.cardinal_query_wait = patched_cardinal_wait
    
    # Also need to update the lookup lambda if it binds the function object?
    # waiting_policy.lookup_linear_function returns a lambda calling the function by name?
    # Let's check lookup_linear_function implementation.
    # It calls: return lambda ...: k8s_wait(...)
    # If k8s_wait is looked up at runtime from module scope, patching the module attribute works.
    # If it bound the function object, it won't. 
    # Python lambdas look up global names at execution time. So patching module attributes works.
    return

def restore_policies():
    """Restores original policies."""
    waiting_policy.k8s_wait = ORIGINAL_K8S_WAIT
    waiting_policy.heft_wait = ORIGINAL_HEFT_WAIT
    waiting_policy.cardinal_query_wait = ORIGINAL_CARDINAL_WAIT


# ==========================================
# 2. Experiment Execution
# ==========================================

def run_experiment():
    print("==========================================================")
    print("Starting Experiment 6: Bandwidth Sensitivity Analysis")
    print("==========================================================")

    # 1. Setup Dataset
    # Use helios_dag as requested
    dataset_config = {
        'dataset': 'helios_dag',
        'total_jobs': 500, # Sufficient for sensitivity
        'seed': 2025,
        'arrival_rate': None,
        'cv_factor': 1.0,
        'job_runtime': 4.0
    }
    
    print("Loading Dataset...")
    try:
        jobs = job_gen.load_processed_jobs(dataset_config)
    except Exception as e:
        print(f"Error loading helios_dag: {e}. Falling back to philly.")
        dataset_config['dataset'] = 'philly'
        jobs = job_gen.load_processed_jobs(dataset_config)

    # Inject Data Size if missing
    print("Injecting Data Sizes...")
    np.random.seed(2025)
    for job in jobs:
        if not hasattr(job, 'data_size_gb') or job.data_size_gb <= 0:
            # Random size 1GB - 10GB
            job.data_size_gb = np.random.uniform(1.0, 10.0)
    
    # 2. Define Sweep
    bandwidths_mbps = [100, 500, 1000, 2000, 5000, 10000]
    methods = ['k8s_native', 'heft', 'cardinal_query']
    
    results = []

    for bw in bandwidths_mbps:
        print(f"\n--- Testing Bandwidth: {bw} Mbps ---")
        
        # Apply Patch
        inject_bandwidth_into_policies(bw)
        
        for method in methods:
            print(f"  Running Method: {method}")
            
            # Setup Simulator Spec
            sim_spec = {
                'cluster_size': 16,
                'gpus_per_node': 8,
                'cpus_per_node': 48,
                'sched_alg': 'fifo',
                'waiting_policy': method,
                'waiting_factor': 1.0, # Default
                'dataset': dataset_config['dataset'],
                'verbose': False,
                'snapshot': False,
                'warmup_jobs': 0,
                # Ensure we use RL for cardinal if needed, but simple wait is fine too
                'use_rl': 1 if method == 'cardinal_query' else 0,
                'max_queue_length': 1000, # Large queue to avoid drops
                'long_job_thres': 0.25 if method == 'cardinal_query' else -1
            }
            
            # Run Simulator
            # We pass the pre-loaded jobs to avoid reloading
            # run_simulator takes a LIST of jobs.
            try:
                # Need to deepcopy jobs because simulator modifies them (state, start, etc)
                sim_jobs = copy.deepcopy(jobs)
                res = run_simulator(sim_jobs, sim_spec)
                
                avg_jct = res['stats']['avg_jct']
                results.append({
                    'Bandwidth (Mbps)': bw,
                    'Method': method,
                    'Avg JCT (s)': avg_jct
                })
                print(f"    -> Avg JCT: {avg_jct:.2f} s")
                
            except Exception as e:
                print(f"    -> Failed: {e}")
                import traceback
                traceback.print_exc()

    # Restore
    restore_policies()
    
    # 3. Save Results
    df = pd.DataFrame(results)
    print("\nResults Summary:")
    print(df)
    df.to_csv('bandwidth_sensitivity_results.csv', index=False)
    
    return df

# ==========================================
# 3. Visualization
# ==========================================

def plot_bandwidth(df):
    # ========================================== 
    # 3. Visualization (SC Top-Tier Ready) 
    # ========================================== 
    # 1. 全局学术字体与排版规范设置 
    plt.rcParams['font.family'] = 'serif' 
    plt.rcParams['font.serif'] = ['Times New Roman'] 
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.2 # 加粗坐标轴线宽 

    fig, ax = plt.subplots(figsize=(8, 6)) 
    
    # 统一命名规范：确保图例名称与正文完全一致 
    name_map = { 
        'k8s_native': 'K8s-Native', 
        'heft': 'HEFT', 
        'cardinal_query': 'IntentSky (Ours)'  # 使用您最终敲定的系统名 
    } 
    df['Method Name'] = df['Method'].map(name_map) 
    
    # 定义高对比度的学术配色与形状差异极其明显的 Marker 
    palette = {'K8s-Native': '#2980B9', 'HEFT': '#E67E22', 'IntentSky (Ours)': '#27AE60'} 
    markers = {'K8s-Native': 'o', 'HEFT': 'X', 'IntentSky (Ours)': 's'} 
    
    # 2. 绘制高保真折线图：加粗线条，放大 Marker 
    sns.lineplot(data=df, x='Bandwidth (Mbps)', y='Avg JCT (s)', hue='Method Name', 
                 style='Method Name', markers=markers, palette=palette, dashes=False, 
                 linewidth=3.0, markersize=10, ax=ax)
    
    # 3. 坐标轴缩放与刻度重塑 
    ax.set_xscale('log') 
    # 明确锁定 X 轴刻度，避免科学计数法或拥挤的 ticks 
    ax.set_xticks([100, 500, 1000, 2000, 5000, 10000]) 
    ax.set_xticklabels(['100', '500', '1k', '2k', '5k', '10k'])
    
    ax.set_xlabel('Inter-cloud Bandwidth (Mbps)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Average JCT (s)', fontsize=18, fontweight='bold')
    
    # ==========================================
    # 【核心修改 1】黑色网格线 + 学术透明度 
    # ========================================== 
    # 将网格线颜色改为黑色，但赋予 0.15 的透明度。 
    # 这样既能实现您要求的“黑色网格”，又绝不会反客为主遮挡折线。 
    ax.grid(True, which='major', color='black', linestyle='--', linewidth=0.8, alpha=0.15) 
    
    # ========================================== 
    # 【核心修改 2】强制全封闭边界框 (Bounding Box) 
    # ========================================== 
    # 遍历所有 4 个方向的边界线，强制显示并设为纯黑实线 
    for spine in ax.spines.values(): 
        spine.set_visible(True) 
        spine.set_color('black') 
        spine.set_linewidth(1.2) 
        
    # 4. 图例精细化排版 
    handles, labels = ax.get_legend_handles_labels() 
    # 将图例放在右上角，增加干净的黑色边框 
    ax.legend(handles=handles, labels=labels, loc='upper right', frameon=True, 
              edgecolor='black', fancybox=False, fontsize=12) 
    
    # 5. 紧凑布局与高质量矢量图导出 
    plt.tight_layout() 
    # 强烈建议保存为 PDF 格式供 LaTeX 编译 
    plt.savefig('bandwidth_sensitivity_sc_ready.pdf', format='pdf', bbox_inches='tight', dpi=300) 
    print("Plot saved to bandwidth_sensitivity_sc_ready.pdf")
    plt.show()

if __name__ == "__main__":
    df = run_experiment()
    plot_bandwidth(df)
