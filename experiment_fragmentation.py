import os
import sys
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skyburst import job_gen

def run_experiment_via_sweep():
    """
    Runs Experiment 3 using the robust run_simulator_sweep.py infrastructure.
    """
    print("Running Experiment 3 (Fragmentationc) via run_simulator_sweep.py...")
    
    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run_simulator_sweep.py')
    log_path = os.path.abspath('logs/fragmentation_experiment.pkl')
    
    # We use 'philly_blocked' which creates the Generator/Encoder mix
    cmd = (
        f"python {script_path} "
        f"--dataset philly_blocked "
        f"--cluster_size 16 "
        f"--sched_alg fifo "
        f"--waiting_policy k8s_native-1 cardinal_query-1 heft-1 infinite-1 "
        f"--use_rl 1 "
        f"--warmup_jobs 0 "
        f"--log {log_path} "
        f"--processes 4 "
        f"--total_jobs 500 "
        f"--verbose"
    )
    
    print(f"Executing: {cmd}")
    # exit_code = os.system(cmd) # Skip re-running if logs exist and recent? 
    # To be safe and ensure new logic, run it.
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        print(f"Error: Simulation failed with exit code {exit_code}")
        return

    print("Simulation completed. Processing results...")
    process_and_plot(log_path)

def process_and_plot(log_path):
    if not os.path.exists(log_path):
        print(f"Error: Log file {log_path} not found.")
        return
        
    with open(log_path, 'rb') as f:
        results = pickle.load(f)
        
    print(f"Loaded {len(results)} simulation results.")
    
    # 1. Regenerate Ground Truth Jobs to get 'gpu_util' and 'mem_util'
    # Config must match what run_simulator_sweep uses for 'philly_blocked'
    print("Regenerating ground truth jobs for attribute mapping...")
    dataset_config = {
        'dataset': 'philly_blocked',
        'total_jobs': 5000, # Matches command line
        'seed': 2026 # Default seed
    }
    # Note: run_simulator_sweep passes args.seed to dataset_config.
    # We used default seed 2025 in sweep script args.
    
    ground_truth_jobs = job_gen.load_processed_jobs(dataset_config)
    job_map = {j.idx: j for j in ground_truth_jobs}
    print(f"Loaded {len(ground_truth_jobs)} jobs with custom attributes.")
    
    all_snapshots = []
    
    GPUS_PER_NODE = 8
    CLUSTER_SIZE = 16
    SNAPSHOT_INTERVAL = 0.5
    
    for res in results:
        spec = res['simulator_spec']
        policy = f"{spec['waiting_policy']}-{spec['waiting_factor']}"
        if 'k8s' in policy: policy_clean = 'K8s-Native'
        # elif 'cardinal' in policy: policy_clean = 'Cardinal-Query'
        elif 'cardinal' in policy: policy_clean = 'IntentSky'
        elif 'heft' in policy: policy_clean = 'HEFT'
        elif 'infinite' in policy: policy_clean = 'Cost-Greedy'
        else: policy_clean = policy
        
        print(f"Processing result for policy: {policy} ({policy_clean})")
        
        # res is a Column-Oriented Dict
        if 'idx' not in res:
            continue
            
        indices = res['idx']
        starts = res['start']
        runtimes = res['runtime']
        allocations = res.get('allocated_gpus', [{} for _ in indices])
        
        # Determine simulation time window
        min_start = min([s for s in starts if s is not None and s >= 0], default=0)
        max_end = max([s + r for s, r in zip(starts, runtimes) if s is not None and s >= 0], default=100)
        
        # Iterate time steps
        current_time = min_start
        
        # Pre-process jobs for fast lookup: List of (start, end, node_id, job_obj)
        active_jobs_info = []
        for i, idx in enumerate(indices):
            if idx not in job_map: continue
            
            start = starts[i]
            if start is None or start < 0: continue # Didn't run
            
            end = start + runtimes[i]
            alloc = allocations[i]
            if not alloc: continue # No allocation (Cloud?)
            
            # Find Node ID (Local)
            # alloc is {node_id: [gpus]}
            node_id = list(alloc.keys())[0]
            if node_id >= CLUSTER_SIZE: continue # Cloud execution
            
            job_obj = job_map[idx]
            active_jobs_info.append({
                'start': start,
                'end': end,
                'node_id': node_id,
                'gpu_util': getattr(job_obj, 'gpu_util', 0),
                'mem_util': getattr(job_obj, 'mem_util', 0)
            })
            
        print(f"  Found {len(active_jobs_info)} valid local jobs.")
        
        # Snapshot Loop
        with tqdm(total=int((max_end - min_start)/SNAPSHOT_INTERVAL)) as pbar:
            while current_time <= max_end:
                # Initialize nodes
                node_stats = {n: {'comp': 0, 'mem': 0} for n in range(CLUSTER_SIZE)}
                
                # Sum util
                active_count = 0
                for info in active_jobs_info:
                    if info['start'] <= current_time < info['end']:
                        nid = info['node_id']
                        if nid in node_stats:
                            node_stats[nid]['comp'] += info['gpu_util']
                            node_stats[nid]['mem'] += info['mem_util']
                            active_count += 1
                
                # Record
                if active_count > 0:
                    for nid, stats in node_stats.items():
                        # Normalize: Sum of utils / Total GPUs
                        # If gpu_util is per-GPU percentage (0-100), then sum / 8 is avg util per GPU.
                        # But we want Node Utilization %.
                        # If a job uses 1 GPU with 100% util, it contributes 100 to sum.
                        # Node has 8 GPUs. Total capacity 800.
                        # So Node Util = Sum / 8.
                        
                        norm_comp = min(100, stats['comp'] / GPUS_PER_NODE)
                        norm_mem = min(100, stats['mem'] / GPUS_PER_NODE)
                        
                        # Only record non-empty nodes to save space
                        if norm_comp > 0 or norm_mem > 0:
                            all_snapshots.append({
                                'timestamp': current_time,
                                'method': policy,
                                'method_clean': policy_clean,
                                'node_id': nid,
                                'compute_util': norm_comp,
                                'mem_util': norm_mem
                            })
                
                current_time += SNAPSHOT_INTERVAL
                pbar.update(1)

    if not all_snapshots:
        print("Error: No snapshot data generated.")
        return
        
    df = pd.DataFrame(all_snapshots)
    df.to_csv('fragmentation_results_reconstructed.csv', index=False)
    print("Saved reconstructed results.")
    
    plot_kde(df)

def plot_kde(df):
    # ========================================== 
    # 1. 顶级学术图表全局设置 (Times New Roman, 线宽) 
    # ========================================== 
    plt.rcParams['font.family'] = 'serif' 
    plt.rcParams['font.serif'] = ['Times New Roman'] 
    plt.rcParams['font.size'] = 14 
    plt.rcParams['axes.linewidth'] = 1.2 
    
    # 使用 sharex 和 sharey 让 4 张图对齐，便于审稿人横向比对 
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True) 
    axes = axes.flatten() 

    methods = ['K8s-Native', 'HEFT', 'Cost-Greedy', 'IntentSky'] 
    
    # 使用更高饱和度与对比度的学术莫兰迪色系 
    colors = {'K8s-Native': 'Blues', 'HEFT': 'Greens', 'Cost-Greedy': 'Purples', 'IntentSky': 'Reds'} 
    point_colors = {'K8s-Native': '#2980B9', 'HEFT': '#27AE60', 'Cost-Greedy': '#8E44AD', 'IntentSky': '#C0392B'} 
    
    for i, method in enumerate(methods): 
        ax = axes[i] 
        data = df[df['method_clean'] == method] 
        
        if not data.empty: 
            # 1. 底层散点：调低 alpha，放在底层 (zorder=1)，展示数据广度但不抢戏 
            sns.scatterplot(data=data, x='compute_util', y='mem_util', 
                            alpha=0.3, color=point_colors[method], s=25, edgecolor='none', ax=ax, zorder=1) 
            
            # 2. KDE 密度等高线：主体填充 (zorder=2) 
            try: 
                sns.kdeplot(data=data, x='compute_util', y='mem_util', 
                            fill=True, cmap=colors[method], thresh=0.05, alpha=0.7, ax=ax, zorder=2) 
                
                # 【高级感技巧】：额外画一层同色系深色等高线勾边，增强 3D 拓扑层次感 
                sns.kdeplot(data=data, x='compute_util', y='mem_util', 
                            levels=4, color=point_colors[method], linewidths=0.8, alpha=0.8, ax=ax, zorder=3) 
            except Exception as e: 
                print(f"KDE failed for {method}: {e}") 
                
            # 3. 目标点标注：使用五角星标注次优边界 
            ax.scatter([80], [80], marker='*', s=250, color='gold', edgecolor='black', linewidth=1,
                       # zorder=10, label='Sub-optimal Boundary (80,80)')
                       zorder=10)

        # 4. 标题与字体设计 
        ax.set_title(f"{method}", fontsize=16, fontweight='bold', pad=10) 
        
        # ========================================== 
        # 【核心修改 1】精准聚焦数据密集区 (40~100) 
        # ========================================== 
        # 稍微拉宽到 35~105，给最边缘的数据点留出一点呼吸空间，防止被边框切断 
        ax.set_xlim(35, 105) 
        ax.set_ylim(35, 105) 
        
        # 显式设定刻度，极其干净 
        ax.set_xticks([40, 60, 80, 100]) 
        ax.set_yticks([40, 60, 80, 100]) 
        
        # 仅在最下排和最左排保留轴标签，减少冗余墨水 
        if i >= 2: 
            ax.set_xlabel("Compute Utilization (%)", fontsize=14, fontweight='bold') 
        else: 
            ax.set_xlabel("") 
            
        if i % 2 == 0: 
            ax.set_ylabel("Memory Utilization (%)", fontsize=14, fontweight='bold') 
        else: 
            ax.set_ylabel("") 

        # ========================================== 
        # 【核心修改 2】封框与深色网格优化 
        # ========================================== 
        # 网格改为黑色，但透明度设为 0.15，虚线。这样既能对齐，又绝对不会遮掩数据 
        ax.grid(True, color='black', linestyle='--', linewidth=0.8, alpha=0.15, zorder=0) 
        
        # 强制四面封框，营造系统论文严谨的“物理容器”感 
        for spine in ax.spines.values(): 
            spine.set_visible(True) 
            spine.set_color('black') 
            spine.set_linewidth(1.2) 

    # ========================================== 
    # 5. 整体排版紧凑化与图例提取 
    # ========================================== 
    handles, labels = ax.get_legend_handles_labels() 
    if handles: 
        # 提取全局唯一图例放于顶部 
        fig.legend(handles[:1], labels[:1], loc='upper center', bbox_to_anchor=(0.5, 1.02), 
                   ncol=1, frameon=True, edgecolor='black', fontsize=12) 

    # 压缩子图之间的空隙，形成紧凑矩阵 
    plt.subplots_adjust(wspace=0.1, hspace=0.25, top=0.92) 
    
    # 强烈建议保存为 PDF 矢量图以供 LaTeX 编译 
    plt.savefig('fig_fragmentation_sc_ready.pdf', format='pdf', bbox_inches='tight', dpi=300) 
    print("Plot saved to fig_fragmentation_sc_ready.pdf")
    plt.show()

if __name__ == "__main__":
    run_experiment_via_sweep()
