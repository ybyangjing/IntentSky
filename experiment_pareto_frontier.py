import argparse
import multiprocessing
import os
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

# Add project root to path to import skyburst modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skyburst import utils
from skyburst.filter_config import apply_filter_config
from run_simulator_sweep import run_grid_search

def main():
    parser = argparse.ArgumentParser(description='Run Pareto Frontier Experiment: SLO vs Cost')
    parser.add_argument('--dataset', type=str, default='helios_dag', help='Dataset to use')
    parser.add_argument('--cluster_size', type=int, default=64, help='Cluster size')
    parser.add_argument('--processes', type=int, default=8, help='Number of parallel processes')
    parser.add_argument('--slo_factor', type=float, default=3.0, help='SLO factor (SLO = Arrival + Factor * Runtime)')
    parser.add_argument('--output_dir', type=str, default='results/pareto_experiment', help='Directory to save results')
    parser.add_argument('--rl_model_path', type=str, default='models/enhanced_a3c_scheduler.pth', help='Path to RL model')
    parser.add_argument('--total_jobs', type=int, default=2000, help='Total number of jobs to simulate')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ==========================================
    # 1. Define Experimental Configurations
    # ==========================================
    
    # Common simulation settings
    common_spec = {
        'dataset': args.dataset,
        'arrival_rate': 32.0, # Explicitly set default instead of None
        'cv_factor': 1.0,
        'total_jobs': args.total_jobs, # Limit jobs for speed if needed, or None for full
        'job_runtime': 4.0,
        'seed': 2026,
        'cluster_size': args.cluster_size,
        'gpus_per_node': 8,
        'cpus_per_node': 48, # or 96 based on simulator default
        'sched_alg': 'fifo',
        'binpack_alg': 'first-fit',
        'backfill': 0,
        'loop': 0, # Default to 0 for most baselines
        'clip_time': 1e9,
        'predict_wait': 0,
        'long_job_thres': 0.25, # Consistent with fig7
        'preempt_cloud_ratio': 3, # Consistent with fig7
        'data_gravity': -1,
        'verbose': False,
        'debug': False,
        'warmup_jobs': 100, # Reduced warmup for faster iteration if needed
        'snapshot': 0,
        'max_queue_length': 1000, # Sufficiently large
        'time_estimator_error': 0,
        # RL Defaults
        'use_rl': 0,
        'train_rl': 0,
        'rl_workers': 1,
        'rl_model_path': args.rl_model_path,
        'rl_save_path': None,
        'rl_lr': 0.0001,
        'rl_gamma': 0.99,
        'rl_entropy_coeff': 0.01,
        'rl_value_coeff': 0.5,
        'rl_max_grad_norm': 0.5,
        'cloud_cost_sensitivity': 1.0, # Default
        'cost_optimization_mode': 1,
        'adaptive_scheduling': 1,
        'multi_objective_weights': [0.4, 0.3, 0.3],
        'enhanced_waiting_policy': 1,
    }

    configs = []

    # --- Baseline 1: K8s-Burst (Zero Wait) ---
    # Performance First: Waits 0s, then bursts to cloud (if filtered or timeout, but zero wait implies immediate try then cloud)
    # Actually, waiting_policy='zero' returns arrival + runtime. 
    # In simulator, if job doesn't fit, it waits. 
    # But user said: "waiting_policy='zero' (Wait 0s, then burst)"
    # To implement "Burst", we usually set max_queue_length small or use filter.
    # But let's follow standard baseline config from fig7: zero-1
    cfg_k8s = common_spec.copy()
    cfg_k8s['waiting_policy'] = 'zero-1'
    cfg_k8s['label'] = 'K8s-Burst'
    cfg_k8s['method_type'] = 'baseline'
    configs.append(cfg_k8s)

    # --- Baseline 2: Cost-Greedy (Infinite Wait) ---
    # Cost First: Wait forever in private cloud
    cfg_greedy = common_spec.copy()
    cfg_greedy['waiting_policy'] = 'infinite-1'
    cfg_greedy['label'] = 'Cost-Greedy'
    cfg_greedy['method_type'] = 'baseline'
    configs.append(cfg_greedy)

    # --- Baseline 3: HEFT (DAG Baseline) ---
    cfg_heft = common_spec.copy()
    cfg_heft['waiting_policy'] = 'heft-1'
    cfg_heft['label'] = 'HEFT'
    cfg_heft['method_type'] = 'baseline'
    configs.append(cfg_heft)

    # --- Baseline 4: Starburst (DAG Baseline) ---
    # Starburst typically uses linear policies.
    # User specified: linear_cost-0.076 and linear_capacity-0.77
    cfg_starburst_1 = common_spec.copy()
    cfg_starburst_1['waiting_policy'] = 'linear_cost-0.076'
    cfg_starburst_1['label'] = 'Starburst (Cost)'
    cfg_starburst_1['method_type'] = 'baseline'
    configs.append(cfg_starburst_1)

    cfg_starburst_2 = common_spec.copy()
    cfg_starburst_2['waiting_policy'] = 'linear_capacity-0.77'
    cfg_starburst_2['label'] = 'Starburst (Cap)'
    cfg_starburst_2['method_type'] = 'baseline'
    configs.append(cfg_starburst_2)
    
    # --- Baseline 5: K8s-Native (Newly Implemented) ---
    cfg_k8s_native = common_spec.copy()
    cfg_k8s_native['waiting_policy'] = 'k8s_native-1'
    cfg_k8s_native['label'] = 'K8s-Native'
    cfg_k8s_native['method_type'] = 'baseline'
    configs.append(cfg_k8s_native)

    # --- Baseline 6: K8s-Constrained (Newly Implemented) ---
    cfg_k8s_constrained = common_spec.copy()
    cfg_k8s_constrained['waiting_policy'] = 'k8s_constrained-1'
    cfg_k8s_constrained['label'] = 'K8s-Constrained'
    cfg_k8s_constrained['method_type'] = 'baseline'
    configs.append(cfg_k8s_constrained)

    # --- Proposed Method: RL (cardinal_query) with Sweep ---
    # Sweep cloud_cost_sensitivity
    sensitivities = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0, 5.0]
    for sens in sensitivities:
        cfg_rl = common_spec.copy()
        cfg_rl['waiting_policy'] = 'cardinal_query-1'
        cfg_rl['use_rl'] = 1
        cfg_rl['cloud_cost_sensitivity'] = sens
        cfg_rl['label'] = f'Proposed (RL)' # Label for grouping
        cfg_rl['sensitivity'] = sens
        cfg_rl['method_type'] = 'proposed'
        configs.append(cfg_rl)

    # Prepare for run_grid_search
    # run_grid_search expects a list of dicts.
    # It requires jobgen_spec to be nested.
    
    final_configs = []
    for cfg in configs:
        # Structure matches run_simulator_sweep.py expectation
        run_cfg = cfg.copy()
        run_cfg['jobgen_spec'] = {
            'dataset': cfg['dataset'],
            'arrival_rate': cfg['arrival_rate'],
            'cv_factor': cfg['cv_factor'],
            'total_jobs': cfg['total_jobs'],
            'job_runtime': cfg['job_runtime'],
            'seed': cfg['seed']
        }
        # Remove flat jobgen fields to avoid confusion (though not strictly necessary if ignored)
        for k in ['dataset', 'arrival_rate', 'cv_factor', 'total_jobs', 'job_runtime', 'seed']:
            if k in run_cfg:
                del run_cfg[k]
        final_configs.append(run_cfg)

    print(f"Total configurations to run: {len(final_configs)}")

    # ==========================================
    # 2. Execution Engine
    # ==========================================
    
    # Run simulations in parallel
    results = run_grid_search(final_configs, num_procs=args.processes)
    
    # ==========================================
    # 3. Metric Extraction Logic
    # ==========================================
    
    parsed_results = []
    
    # Find K8s-Burst cost for normalization
    k8s_burst_cost = 1.0
    
    # First pass: map results to configs and extract raw metrics
    temp_results = []
    
    for i, res in enumerate(results):
        cfg = configs[i]
        
        jobs_arrival = res['arrival']
        jobs_start = res['start']
        jobs_runtime = res['runtime']
        
        # Calculate waiting times
        waiting_times = jobs_start - jobs_arrival
        
        # Check violations
        # Violation condition: waiting_time > (args.slo_factor - 1) * jobs_runtime
        thresholds = (args.slo_factor - 1.0) * jobs_runtime
        violations = waiting_times > thresholds
        violation_count = np.sum(violations)
        total_jobs = len(jobs_arrival)
        slo_rate = violation_count / total_jobs if total_jobs > 0 else 0
        
        # Operational Cost = Total Cloud Cost
        total_cost = res['stats']['total_cloud_cost']
        
        entry = {
            'label': cfg['label'],
            'method_type': cfg['method_type'],
            'sensitivity': cfg.get('sensitivity', None),
            'slo_violation_rate': slo_rate,
            'total_cost': total_cost,
            'waiting_policy': cfg['waiting_policy']
        }
        temp_results.append(entry)
        
        if cfg['label'] == 'K8s-Burst':
            k8s_burst_cost = total_cost

    if k8s_burst_cost == 0:
        k8s_burst_cost = 1.0 

    # Normalize and build final list
    for entry in temp_results:
        entry['normalized_cost'] = entry['total_cost'] / k8s_burst_cost
        parsed_results.append(entry)

    # Save raw data
    df = pd.DataFrame(parsed_results)
    csv_path = os.path.join(args.output_dir, 'pareto_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    print(tabulate(df, headers='keys', tablefmt='psql'))

    # ==========================================
    # 4. Visualization
    # ==========================================
    plot_pareto(df, args.output_dir)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import seaborn as sns
import os

def plot_pareto(df, output_dir):
    """
    Publication-Quality Pareto Frontier Plot
    Optimized for Top-Tier Systems Conferences (SC, OSDI, HPCA)
    """
    print("Generating High-Fidelity Pareto Frontier Plot...")

    # ==========================================
    # 1. Global Style Setup (IEEE/ACM Double Column)
    # ==========================================
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.axisbelow'] = True
    
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    # Grid styling
    ax.grid(True, linestyle=':', color='gray', alpha=0.5, zorder=0)

    # ==========================================
    # 2. Design System: Colors & Markers
    # ==========================================
    # Use distinct, color-blind friendly colors and high-contrast markers
    styles = {
        'IntentSky': {'color': '#c0392b', 'marker': '*', 's': 350, 'edgecolor': 'black', 'lw': 1.0},
        'Cost-Greedy': {'color': '#2980b9', 'marker': 'o', 's': 150, 'edgecolor': 'black', 'lw': 1.2},
        'HEFT': {'color': '#f39c12', 'marker': 's', 's': 130, 'edgecolor': 'black', 'lw': 1.2},
        'K8s-Native': {'color': '#8e44ad', 'marker': 'v', 's': 130, 'edgecolor': 'black', 'lw': 1.2},
        'K8s-Burst': {'color': '#27ae60', 'marker': '^', 's': 130, 'edgecolor': 'black', 'lw': 1.2},
        'K8s-Constrained': {'color': '#16a085', 'marker': '<', 's': 130, 'edgecolor': 'black', 'lw': 1.2},
        'Starburst (Cost)': {'color': '#d35400', 'marker': 'X', 's': 130, 'edgecolor': 'black', 'lw': 1.2},
        'Starburst (Cap)': {'color': '#7f8c8d', 'marker': 'D', 's': 110, 'edgecolor': 'black', 'lw': 1.2},
    }

    # ==========================================
    # 3. Draw Optimal Region (Background)
    # ==========================================
    # Shaded green box for the ideal Pareto area
    opt_region = patches.Rectangle((-0.05, -0.01), 0.45, 0.06, 
                                   linewidth=0, facecolor='#e8f5e9', alpha=0.7, zorder=1)
    ax.add_patch(opt_region)
    ax.text(0.12, 0.01, 'Optimal Region', color='#27ae60', fontstyle='italic', 
            fontsize=13, fontweight='bold', alpha=0.8, zorder=2)

    # ==========================================
    # 4. Plot Data Points & Annotations
    # ==========================================
    bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.5, alpha=0.9)
    
    # Store handles for the legend
    legend_handles = []
    
    for _, row in df.iterrows():
        label = row['label']
        # Map proposed RL to IntentSky
        display_label = 'IntentSky' if label.startswith('Proposed') else label
        
        # Fallback style if missing
        st = styles.get(display_label, {'color': 'gray', 'marker': 'o', 's': 100, 'edgecolor': 'black', 'lw': 1})
        
        scatter = ax.scatter(row['normalized_cost'], row['slo_violation_rate'], 
                             color=st['color'], marker=st['marker'], s=st['s'], 
                             edgecolors=st['edgecolor'], linewidths=st['lw'], 
                             label=display_label, zorder=4)
        
        if display_label not in [h.get_label() for h in legend_handles]:
            legend_handles.append(scatter)

        # Smart Annotation Placement (Avoid overlaps)
        text_str = f"{display_label}\n(C={row['normalized_cost']:.2f}, S={row['slo_violation_rate']*100:.1f}%)"
        
        if display_label == 'IntentSky':
            ax.annotate(text_str, (row['normalized_cost'], row['slo_violation_rate']), 
                        xytext=(-10, -35), textcoords='offset points', ha='center', 
                        fontsize=10, color='#900C3F', fontweight='bold', bbox=bbox_props, 
                        arrowprops=dict(arrowstyle="->", color='#c0392b', lw=1.5), zorder=5)
        elif display_label == 'Cost-Greedy':
            ax.annotate(text_str, (row['normalized_cost'], row['slo_violation_rate']), 
                        xytext=(15, 10), textcoords='offset points', ha='left', 
                        fontsize=10, color='black', bbox=bbox_props, zorder=5)
        elif display_label == 'HEFT':
            ax.annotate(text_str, (row['normalized_cost'], row['slo_violation_rate']), 
                        xytext=(-20, 15), textcoords='offset points', ha='center', 
                        fontsize=10, color='black', bbox=bbox_props, zorder=5)

    # Note: K8s and Starburst variants are usually clustered around x=1.0.
    # We group their label to avoid a mess.
    ax.annotate("Scalar / Heuristic Schedulers\nCluster (C ≈ 1.0, S ≈ 0%)", 
                xy=(1.0, 0.0), xytext=(0.85, 0.03), 
                fontsize=10, bbox=bbox_props, ha='center', 
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color='gray', lw=1.5), zorder=5)

    # ==========================================
    # 5. Inset Axes (Zoom: Optimal Region) - REMOVED
    # ==========================================
    # Move the inset to the middle-left, just above the optimal region
    # axins = ax.inset_axes([0.35, 0.45, 0.35, 0.35])
    
    # for _, row in df.iterrows():
    #     label = row['label']
    #     display_label = 'IntentSky' if label.startswith('Proposed') else label
    #     if display_label in ['IntentSky', 'Cost-Greedy']:
    #         st = styles[display_label]
    #         axins.scatter(row['normalized_cost'], row['slo_violation_rate'], 
    #                       color=st['color'], marker=st['marker'], s=st['s']*0.8, 
    #                       edgecolors=st['edgecolor'], linewidths=st['lw'], zorder=4)

    # # Set zoom limits exactly targeting the optimal region and Cost-Greedy
    # axins.set_xlim(0.18, 0.35)
    # axins.set_ylim(-0.005, 0.05)
    # axins.set_title("Zoom: Optimal Region", fontsize=11, fontweight='bold', pad=5)
    # axins.grid(True, linestyle=':', alpha=0.5)
    # axins.tick_params(labelsize=9)
    
    # # Draw magical connecting lines
    # mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="gray", lw=1.0, alpha=0.6, linestyle='--')

    # ==========================================
    # 6. Compaction & Final Polish
    # ==========================================
    # COMPACT AXES: Remove dead space!
    ax.set_xlim(-0.02, 1.1)  # Cut off right side blank space
    ax.set_ylim(-0.01, 0.08) # Cut off top blank space
    
    ax.set_xlabel('Normalized Operational Cost (Lower is Better)', fontsize=13, fontweight='bold')
    ax.set_ylabel('SLO Violation Rate (Lower is Better)', fontsize=13, fontweight='bold')

    # Integrated Legend inside the plot (Top Right)
    legend = ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                       ncol=2, framealpha=0.9, edgecolor='gray', fontsize=10)
    legend.set_zorder(10)

    # Box borders (Closed Frame)
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.2)

    plt.tight_layout()
    
    # Save high-res
    png_path = os.path.join(output_dir, 'pareto_frontier_top_tier.png')
    pdf_path = os.path.join(output_dir, 'pareto_frontier_top_tier.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Publication-ready plots saved to:\n{png_path}\n{pdf_path}")
    plt.close()

if __name__ == '__main__':
    main()
