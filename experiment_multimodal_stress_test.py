
import argparse
import os
import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skyburst.traces import helios
from skyburst.job import Job
from skyburst import run_simulator
from skyburst import utils
from skyburst.traces import dag  # For CACHE_DIR

# --- Constants ---
TOTAL_GPU_JOBS = 2000
RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# RATIOS = [0.0, 0.5, 1.0] # For faster testing
SEEDS_START = 2026

# --- 1. Workload Mixer ---

def generate_mixed_workload(ratio, total_gpu_jobs=2000, seed=2026):
    """
    Generates a workload with a specific ratio of Multimodal jobs.
    
    Strategy to maintain Load Balance:
    1. Select `total_gpu_jobs` from the source trace.
    2. Randomly shuffle them.
    3. The first `total_gpu_jobs * ratio` jobs are transformed into Multimodal DAGs (Encoder + Generator).
    4. The remaining jobs are kept as Unimodal (Single).
    
    This ensures that the total GPU demand (sum of GPU * runtime) is statistically identical 
    across all ratios, because the "Generator" part of the DAG is the original GPU job.
    The only variable is the addition of CPU-only parent tasks and dependencies.
    """
    # Load raw Helios traces
    # We load from the integrated CSV found in the root directory
    csv_path = r'E:\starburst_Relsing_Sky\integrated_helios_workload.csv'
    if not os.path.exists(csv_path):
        # Fallback to current directory or parent
        if os.path.exists('../integrated_helios_workload.csv'):
            csv_path = '../integrated_helios_workload.csv'
        elif os.path.exists('integrated_helios_workload.csv'):
            csv_path = 'integrated_helios_workload.csv'
        else:
            raise FileNotFoundError(f"Could not find integrated_helios_workload.csv in {os.getcwd()} or E:\\starburst_Relsing_Sky")

    print(f"Loading traces from {csv_path}...")
    try:
        # Use pandas for speed
        df = pd.read_csv(csv_path)
        
        # Filter for valid GPU jobs to serve as the "Base" for our workload
        # We only want jobs that actually use GPUs, as these are the ones we'll either 
        # keep as Single (Unimodal) or prefix with an Encoder (Multimodal).
        # We ignore existing DAG structure in the CSV to enforce our own Ratio.
        valid_df = df[
            (df['gpu_num'] > 0) & 
            (df['duration'] > 0) &
            (df['state'].isin(['COMPLETED', 'TIMEOUT', 'Pass']))
        ]
        
        raw_jobs = valid_df.to_dict('records')
        print(f"Found {len(raw_jobs)} valid GPU jobs.")
        
    except Exception as e:
        print(f"Error loading traces: {e}")
        raise e

    # We need a fixed subset of jobs to ensure fairness across ratios
    # So we seed the selection of the "Base Pool"
    np.random.seed(seed) # Fixed seed for base pool selection
    
    # Randomly sample TOTAL_GPU_JOBS from the valid jobs
    if len(raw_jobs) < total_gpu_jobs:
        print(f"Warning: Not enough jobs in trace ({len(raw_jobs)}). Using all.")
        selected_indices = np.arange(len(raw_jobs))
    else:
        selected_indices = np.random.choice(len(raw_jobs), size=total_gpu_jobs, replace=False)
    
    base_pool = [raw_jobs[i] for i in selected_indices]
    
    # Sort by submission time to respect arrival order
    # Helper to parse time
    def parse_time(t_str):
        try:
            return pd.to_datetime(t_str)
        except:
            return pd.to_datetime('2020-01-01') # Fallback

    base_pool.sort(key=lambda x: parse_time(x['submit_time']))
    
    # Normalize Arrival Times (start from 0)
    start_time = parse_time(base_pool[0]['submit_time'])
    
    # --- LOAD COMPRESSION LOGIC ---
    # We need to compress the arrival times to create a high-load scenario (Stress Test).
    # Otherwise, sampling 2000 jobs from a long trace results in empty clusters.
    
    # 1. Calculate Total Work (GPU-Hours)
    total_gpu_hours = sum([float(j['duration'])/3600.0 * int(j['gpu_num']) for j in base_pool])
    
    # 2. Cluster Capacity
    cluster_gpus = 64 * 8 # 512 GPUs
    target_load = 0.95 # Target high load
    
    # 3. Calculate Target Duration (Hours) to achieve Target Load
    # Load = Work / (Capacity * Duration) -> Duration = Work / (Capacity * Load)
    target_duration = total_gpu_hours / (cluster_gpus * target_load)
    
    # 4. Original Duration
    original_end_time = parse_time(base_pool[-1]['submit_time'])
    original_duration = (original_end_time - start_time).total_seconds() / 3600.0
    
    if original_duration == 0: original_duration = 1.0 # Safety
    
    compression_factor = target_duration / original_duration
    print(f"Compressing workload by factor {compression_factor:.6f} to achieve ~{target_load} load.")
    
    # Now, determine which ones become Multimodal based on Ratio
    # We shuffle indices to randomly assign "Multimodal Status"
    # But wait, we want the "Total Load" to be comparable. 
    # If we shuffle, it's statistically comparable.
    num_multimodal = int(total_gpu_jobs * ratio)
    
    is_multimodal_mask = np.zeros(len(base_pool), dtype=bool)
    is_multimodal_mask[:num_multimodal] = True
    np.random.shuffle(is_multimodal_mask) # Shuffle who gets to be multimodal
    
    final_jobs = []
    current_new_id = 0
    
    # Process
    for i, raw_job in enumerate(base_pool):
        # Common attributes
        submit_time = parse_time(raw_job['submit_time'])
        raw_arrival = (submit_time - start_time).total_seconds() / 3600.0
        
        # Apply Compression
        arrival = raw_arrival * compression_factor
        
        runtime = float(raw_job['duration']) / 3600.0 # Convert seconds to hours
        
        # Resources
        resources = {'GPUs': int(raw_job['gpu_num']), 'CPUs': int(raw_job['cpu_num'])}
        # Cost (approximate)
        cost = (resources['GPUs'] + resources['CPUs'] / 53.0) * runtime
        
        # Nodes (default to 1 if missing)
        nodes = int(raw_job.get('node_num', 1))
        
        if is_multimodal_mask[i]:
            # === Multimodal DAG (Parent -> Child) ===
            
            # 1. Parent (Encoder) - CPU Only
            parent_duration = max(1.0/60.0, runtime * 0.2) # 20% of child, min 1 min
            parent_res = {'CPUs': 4, 'GPUs': 0}
            parent_cost = 4 * parent_duration
            
            parent_job = Job(
                idx=current_new_id,
                arrival=arrival,
                runtime=parent_duration,
                resources=parent_res,
                cost=parent_cost
            )
            parent_job.job_type = 'Encoder'
            parent_job.target_cloud = 'Private'
            parent_job.data_size = 0.1
            parent_job.dependencies = []
            
            final_jobs.append(parent_job)
            parent_id = current_new_id
            current_new_id += 1
            
            # 2. Child (Generator) - The original GPU job
            child_job = Job(
                idx=current_new_id,
                arrival=arrival, # Arrives same time, but dependent
                runtime=runtime,
                resources=resources,
                cost=cost
            )
            child_job.job_type = 'Generator'
            child_job.target_cloud = 'Public'
            child_job.data_size = np.random.uniform(5.0, 20.0)
            child_job.dependencies = [parent_id]
            child_job.dependency_parent = parent_id
            child_job.nodes = nodes # Important
            
            final_jobs.append(child_job)
            current_new_id += 1
            
        else:
            # === Unimodal (Single) ===
            job = Job(
                idx=current_new_id,
                arrival=arrival,
                runtime=runtime,
                resources=resources,
                cost=cost
            )
            job.job_type = 'Single'
            job.target_cloud = 'Any'
            job.data_size = np.random.uniform(0.1, 1.0)
            job.dependencies = []
            job.nodes = nodes
            
            final_jobs.append(job)
            current_new_id += 1
            
    # Sort by arrival
    final_jobs.sort(key=lambda j: j.arrival)
    
    # Re-index to be safe (0 to N-1)
    id_map = {}
    remapped_jobs = []
    for new_idx, job in enumerate(final_jobs):
        id_map[job.idx] = new_idx
        job.idx = new_idx
        remapped_jobs.append(job)
        
    # Fix dependencies
    for job in remapped_jobs:
        if job.dependencies:
            job.dependencies = [id_map[d] for d in job.dependencies]
        if job.dependency_parent is not None:
            job.dependency_parent = id_map[job.dependency_parent]
            
    return remapped_jobs

# --- 2. Simulation Runner ---

def generate_data_run_simulator(run_config):
    # This wrapper is needed for multiprocessing
    # We bypass job_gen.load_processed_jobs by pre-loading data?
    # No, we use the cache trick.
    
    # The run_config contains 'jobgen_spec'.
    # jobgen_spec has 'dataset' and 'seed'.
    # The simulator calls job_gen.load_processed_jobs(jobgen_spec)
    # which calls dag.load_dag_traces_with_cache(jobgen_spec)
    # which looks for dag_{dataset}_seed{seed}_v2.pkl
    
    # So we don't need to change this function, just ensure the file exists.
    from skyburst import job_gen
    proc_jobs = job_gen.load_processed_jobs(run_config['jobgen_spec'])
    return run_simulator(proc_jobs, run_config)

def run_stress_test():
    results_data = []
    
    # Ensure cache dir exists
    if not os.path.exists(dag.CACHE_DIR):
        os.makedirs(dag.CACHE_DIR)

    # Prepare common spec
    common_spec = {
        'cluster_size': 64,
        'gpus_per_node': 8,
        'cpus_per_node': 48,
        'sched_alg': 'fifo',
        'binpack_alg': 'first-fit',
        'backfill': 0,
        'loop': 0,
        'clip_time': 1e9,
        'predict_wait': 0,
        'long_job_thres': -1,
        'preempt_cloud_ratio': -1,
        'data_gravity': -1,
        'verbose': False,
        'debug': False,
        'warmup_jobs': 200, # Smaller warmup for this smaller workload
        'snapshot': 0,
        'max_queue_length': -1,
        'time_estimator_error': 0,
        
        # RL Configs
        'use_rl': 0,
        'train_rl': 0,
        'rl_workers': 1,
        'rl_model_path': 'models/enhanced_a3c_scheduler.pth', # Default
        'rl_save_path': None,
        'rl_lr': 0.0003,
        'rl_gamma': 0.995,
        'rl_entropy_coeff': 0.01,
        'rl_value_coeff': 0.5,
        'rl_max_grad_norm': 0.5,
        'cloud_cost_sensitivity': 0.5, # Fixed for stress test
        'cost_optimization_mode': 1,
        'adaptive_scheduling': 1,
        'multi_objective_weights': [0.4, 0.3, 0.3],
        'enhanced_waiting_policy': 1,
    }

    # Define Policies
    policies = []
    
    # 1. Proposed (RL)
    p_rl = common_spec.copy()
    p_rl['waiting_policy'] = 'cardinal_query'
    p_rl['use_rl'] = 1
    p_rl['label'] = 'RelsingSky'
    p_rl['method_type'] = 'proposed'
    # Important: Point to a valid model checkpoint if available.
    # Assuming 'models/enhanced_a3c_scheduler.pth' exists or will be ignored if not training?
    # Actually, for inference, it needs to exist. 
    # If not, the simulator might fail or initialize random.
    # We'll assume the user has a model or we run in a mode that works.
    policies.append(p_rl)
    
    # 2. K8s-Burst
    p_k8s = common_spec.copy()
    p_k8s['waiting_policy'] = 'zero-1'
    p_k8s['label'] = 'K8s-Burst'
    p_k8s['method_type'] = 'baseline'
    policies.append(p_k8s)
    
    # 3. HEFT
    p_heft = common_spec.copy()
    p_heft['waiting_policy'] = 'heft-1'
    p_heft['label'] = 'HEFT'
    p_heft['method_type'] = 'baseline'
    policies.append(p_heft)
    
    # 4. Starburst (Baselines)
    # Starburst (Cost)
    cfg_starburst_1 = common_spec.copy()
    cfg_starburst_1['waiting_policy'] = 'linear_cost-0.076'
    cfg_starburst_1['label'] = 'Starburst (Cost)'
    cfg_starburst_1['method_type'] = 'baseline'
    policies.append(cfg_starburst_1)

    # Starburst (Cap)
    cfg_starburst_2 = common_spec.copy()
    cfg_starburst_2['waiting_policy'] = 'linear_capacity-0.77'
    cfg_starburst_2['label'] = 'Starburst (Cap)'
    cfg_starburst_2['method_type'] = 'baseline'
    policies.append(cfg_starburst_2)

    # 5. Cost-Greedy
    p_greedy = common_spec.copy()
    p_greedy['waiting_policy'] = 'infinite-1'
    p_greedy['label'] = 'Cost-Greedy'
    p_greedy['method_type'] = 'baseline'
    policies.append(p_greedy)
    
    # 5. K8s-Native
    p_k8s_nat = common_spec.copy()
    p_k8s_nat['waiting_policy'] = 'k8s_native-1'
    p_k8s_nat['label'] = 'K8s-Native'
    p_k8s_nat['method_type'] = 'baseline'
    policies.append(p_k8s_nat)

    # 6. K8s-Constrained
    p_k8s_const = common_spec.copy()
    p_k8s_const['waiting_policy'] = 'k8s_constrained-1'
    p_k8s_const['label'] = 'K8s-Constrained'
    p_k8s_const['method_type'] = 'baseline'
    policies.append(p_k8s_const)

    # Loop Ratios
    for ratio in RATIOS:
        # Generate Seed
        seed = SEEDS_START + int(ratio * 100)
        
        # 1. Generate Workload
        workload = generate_mixed_workload(ratio, total_gpu_jobs=TOTAL_GPU_JOBS, seed=2024)
        
        # 2. Save to Cache
        # Format: dag_{dataset}_seed{seed}_v2.pkl
        # We use dataset='helios_dag'
        cache_filename = f"dag_helios_dag_seed{seed}_v2.pkl"
        cache_path = os.path.join(dag.CACHE_DIR, cache_filename)
        
        print(f"Saving workload to {cache_path}...")
        with open(cache_path, 'wb') as f:
            pickle.dump(workload, f)
            
        # 3. Prepare Configs for this Ratio
        run_configs = []
        for pol in policies:
            cfg = pol.copy()
            cfg['jobgen_spec'] = {
                'dataset': 'helios_dag',
                'seed': seed, # This triggers loading the file we just saved
                'arrival_rate': None, # Ignored for DAG loading
                'cv_factor': None,
                'total_jobs': None,
                'job_runtime': None
            }
            # Add metadata for results
            cfg['multimodal_ratio'] = ratio
            run_configs.append(cfg)
            
        # 4. Run Simulator
        # Use multiprocessing pool
        with multiprocessing.Pool(processes=6) as pool:
            # We need to wrap config in list for starmap
            args_list = [[r] for r in run_configs]
            batch_results = pool.starmap(generate_data_run_simulator, args_list)
            
        # 5. Collect Metrics
        for cfg, res in zip(run_configs, batch_results):
            # Extract metrics
            stats = res['stats']
            avg_jct = stats['avg_jct']
            avg_cost = stats.get('avg_cloud_cost', 0)
            
            # Record
            record = {
                'ratio': ratio,
                'method': cfg['label'],
                'avg_jct': avg_jct,
                'avg_cost': avg_cost,
                'makespan': 0 # Not strictly needed for this plot
            }
            results_data.append(record)
            print(f"Finished: Ratio={ratio}, Method={cfg['label']}, JCT={avg_jct:.2f}")

    # --- 3. Save Results ---
    df = pd.DataFrame(results_data)
    csv_path = 'stress_test_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # --- 4. Plotting ---
    plot_stress_test(df)

def plot_stress_test(df):
    print("Plotting results...")
    
    # Normalize JCT
    # Normalize against K8s-Burst at Ratio=0.0? Or Proposed at Ratio=0.0?
    # User said: "Divide by the JCT of the "Proposed" method at Ratio=0% (or normalize against K8s at Ratio=0%)"
    # Let's normalize against K8s-Burst at Ratio=0.0 for a standard baseline reference.
    
    baseline_row = df[(df['ratio'] == 0.0) & (df['method'] == 'K8s-Burst')]
    if not baseline_row.empty:
        norm_factor = baseline_row['avg_jct'].values[0]
    else:
        # Fallback
        norm_factor = df['avg_jct'].min()
        
    df['normalized_jct'] = df['avg_jct'] / norm_factor
    
    # Plot
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.4)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    # Markers mapping
    markers = {
        'RelsingSky': '*', 
        'K8s-Burst': 'o', 
        'HEFT': '^', 
        'Cost-Greedy': 's', 
        'K8s-Native': 'D', 
        'K8s-Constrained': 'v',
        'Starburst (Cost)': 'X',
        'Starburst (Cap)': 'P'
    }
    colors = {
        'RelsingSky': '#d62728', 
        'K8s-Burst': '#1f77b4', 
        'HEFT': '#ff7f0e', 
        'Cost-Greedy': '#2ca02c', 
        'K8s-Native': '#9467bd', 
        'K8s-Constrained': '#8c564b',
        'Starburst (Cost)': '#17becf',
        'Starburst (Cap)': '#e377c2'
    }
    
    methods = df['method'].unique()
    
    for method in methods:
        subset = df[df['method'] == method].sort_values('ratio')
        plt.plot(subset['ratio'], subset['normalized_jct'], 
                 marker=markers.get(method, 'o'), 
                 color=colors.get(method, 'gray'),
                 label=method, linewidth=2.5, markersize=8)
                 
    plt.xlabel('Multimodal Task Ratio', fontweight='bold')
    plt.ylabel('Normalized Avg. JCT', fontweight='bold')
    plt.title('Experiment 2: Multimodal Robustness Stress Test', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig('stress_test_results.pdf', bbox_inches='tight')
    plt.savefig('stress_test_results.png', bbox_inches='tight')
    print("Plot saved to stress_test_results.pdf/png")

if __name__ == '__main__':
    run_stress_test()
