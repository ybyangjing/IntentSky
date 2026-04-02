import argparse
import itertools
import json
import multiprocessing
import pickle
import os
import time
import torch

from skyburst import job_gen, run_simulator
from skyburst import utils
from skyburst.filter_config import apply_filter_config

# GPU设备检测和配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"模拟器扫描模块使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU设备名称: {torch.cuda.get_device_name()}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.backends.cudnn.benchmark = True  # 优化GPU性能
    # 设置多进程GPU使用策略
    torch.multiprocessing.set_start_method('spawn', force=True)


def generate_data_run_simulator(run_config):
    proc_jobs = job_gen.load_processed_jobs(
        dataset_config=run_config['jobgen_spec'])
    
    return run_simulator(proc_jobs, run_config)


def run_grid_search(run_configs, num_procs=32):
    for i, r in enumerate(run_configs):
        r['pbar_idx'] = i
    run_configs = [[r] for r in run_configs]
    with multiprocessing.Pool(processes=num_procs) as pool:
        results = pool.starmap(generate_data_run_simulator, run_configs)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'Run a hyperparameter sweep over diff. values in a hybrid cloud simulator.'
    )

    # Arguments for Data Generation
    parser.add_argument("--dataset",
                        type=str,
                        choices=[
                            "philly", "philly_gen", "gen_gpu", "helios",
                            "synthetic", "helios_gen", "philly_dag", "helios_dag", "helios_blocked", "philly_blocked", "philly_privacy"
                        ],
                        default='philly',
                        help='Choose dataset to run simulator from.')
    parser.add_argument('--arrival_rate',
                        type=float,
                        nargs='+',
                        default=None,
                        help='Arrival rate for generated jobs.')
    parser.add_argument('--cv_factor',
                        type=float,
                        nargs='+',
                        default=1.0,
                        help='Varies job burstiness.')
    parser.add_argument('--total_jobs',
                        type=int,
                        default=None,
                        help='How many jobs should be generated.')
    parser.add_argument('--job_runtime',
                        type=float,
                        default=4.0,
                        help='Average runtime for job.')

    # Arguments for Cluster specifications.
    parser.add_argument('--cluster_size',
                        type=int,
                        nargs='+',
                        default=64,
                        help='Size of the cluster (i.e. # of cluster nodes)')
    parser.add_argument('--gpus_per_node',
                        type=int,
                        default=8,
                        help='Number of GPU(s) per cluster node')
    parser.add_argument('--cpus_per_node',
                        type=int,
                        default=48,
                        help='Number of CPU(s) per cluster node')

    # Arguments for Policy specifications.
    parser.add_argument(
        '--sched_alg',
        type=str,
        nargs='+',
        default='fifo',
        help='Scheduling algorithm specifying order of the queue.')
    parser.add_argument('--binpack_alg',
        type=str,
        nargs='+',
        default='first-fit',
        choices=['first-fit', 'best-fit', 'worst-fit'],
        help='Binpacking algorithm for the cluster.')
    parser.add_argument(
        '--waiting_policy',
        type=str,
        nargs='+',
        default='cardinal_cocn',
        help='Waiting policy (how long jobs should at max wait in the cloud).')
    parser.add_argument('--clip_time',
                        type=float,
                        default=1e9,
                        nargs='+',
                        help='Sets maximum clipping time for a job.')
    parser.add_argument('--backfill',
                        type=int,
                        nargs='+',
                        default=0,
                        choices=[0, 1],
                        help='Enable backfill (assumes time estimator)')
    parser.add_argument(
        '--loop',
        type=int,
        nargs='+',
        default=0,
        choices=[0, 1],
        help=
        'Enable loop scheduling (just loop through entire queue, remove HoL)')
    parser.add_argument(
        '--predict_wait',
        type=int,
        nargs='+',
        default=0,
        choices=[0, 1, 2],
        help=
        'Enable prediction. (Jobs predict if they can be assigned to cluster before timing out)'
    )
    parser.add_argument('--time_estimator_error',
                        type=int,
                        nargs='+',
                        default=0,
                        help='Time estimator error')
    parser.add_argument('--max_queue_length',
                        type=int,
                        default=-1,
                        nargs='+',
                        help='Sets maximum length for queue.')

    parser.add_argument(
        '--long_job_thres',
        type=float,
        nargs='+',
        default=-1,
        help='Long job threshold (if lower than threshold move to cloud).')
    parser.add_argument('--preempt_cloud_ratio',
                        type=float,
                        nargs='+',
                        default=-1,
                        help='Cloud preemption threshold.')
    parser.add_argument('--data_gravity',
                        type=float,
                        nargs='+',
                        default=-1,
                        help='Data gravity delay for running in the cloud.')

    parser.add_argument('--seed',
                        type=int,
                        default=2025,
                        help='Seed for data generation.')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Prints out simulator state at every timestep')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Appends python debugger at every timestemp')
    parser.add_argument(
        '--warmup_jobs',
        type=int,
        default=5000,
        help=
        'Jobs to not consider for final metrics at the beg. and end. of simulator'
    )
    parser.add_argument(
        '--filter_name',
        type=str,
        default=None,
        help='Specifies filter config.')
    parser.add_argument(
        '--log',
        type=str,
        default='logs/philly_end2end.log',
        help='Specifies where to save the simulator sweep results.')

    parser.add_argument(
        '--snapshot',
        type=int,
        default=0,
        choices=[0, 1],
        help=
        'Specifies whether to save queue state at the end of each iteration. (This is used for underutilization analysis.)'
    )
    parser.add_argument(
        '--snapshot_interval',
        type=float,
        default=-1,
        help='Interval for taking node utilization snapshots (Experiment 3).'
    )

    parser.add_argument('--use_rl',
                        type=int,
                        default=1,
                        choices=[0, 1],
                        help='使用强化学习进行调度决策')
    parser.add_argument('--train_rl',
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help='是否训练强化学习模型')
    parser.add_argument('--rl_workers',
                        type=int,
                        default=8,
                        help='A3C工作线程数量')
    parser.add_argument('--rl_model_path',
                        type=str,
                        default='models/enhanced_a3c_scheduler.pth',
                        help='预训练RL模型路径')
    parser.add_argument('--rl_save_path',
                        type=str,
                        default='models/enhanced_a3c_scheduler.pth',
                        help='保存RL模型的路径')
    parser.add_argument('--rl_lr',
                        type=float,
                        default=0.0003,
                        help='RL学习率')
    parser.add_argument('--rl_gamma',
                        type=float,
                        default=0.995,
                        help='RL折扣因子')
    parser.add_argument('--rl_entropy_coeff',
                        type=float,
                        default=0.01,
                        help='RL熵系数，用于鼓励探索')
    parser.add_argument('--rl_value_coeff',
                        type=float,
                        default=0.5,
                        help='RL价值损失系数')
    parser.add_argument('--rl_max_grad_norm',
                        type=float,
                        default=0.5,
                        help='RL梯度裁剪阈值')
    parser.add_argument('--cloud_cost_sensitivity',
                        type=float,
                        default=1.5,
                        help='云成本敏感性倍数')
    parser.add_argument('--cost_optimization_mode',
                        type=int,
                        default=1,
                        choices=[0, 1],
                        help='是否启用成本优化模式')
    parser.add_argument('--adaptive_scheduling',
                        type=int,
                        default=1,
                        choices=[0, 1],
                        help='是否启用自适应调度')
    parser.add_argument('--multi_objective_weights',
                        type=str,
                        default='0.4,0.3,0.3',
                        help='多目标权重：成本,利用率,完成时间')
    parser.add_argument('--enhanced_waiting_policy',
                        type=int,
                        default=1,
                        choices=[0, 1],
                        help='是否使用增强的等待时间预测器')
    parser.add_argument('--processes',
                        type=int,
                        default=8,
                        help='并行进程数')

    args = parser.parse_args()
    grid_search_config = {
        # Cluster config
        'cluster_size': args.cluster_size,
        'gpus_per_node': args.gpus_per_node,
        'cpus_per_node': args.cpus_per_node,
        # Policy config
        'sched_alg': args.sched_alg,
        'binpack_alg': args.binpack_alg,
        'waiting_policy': args.waiting_policy,
        'backfill': args.backfill,
        'loop': args.loop,
        'clip_time': args.clip_time,
        'predict_wait': args.predict_wait,
        'long_job_thres': args.long_job_thres,
        'preempt_cloud_ratio': args.preempt_cloud_ratio,
        'data_gravity': args.data_gravity,
        # Simulator config
        'verbose': args.verbose,
        'debug': args.debug,
        'warmup_jobs': args.warmup_jobs,
        'snapshot': args.snapshot,
        'snapshot_interval': args.snapshot_interval,
        'max_queue_length': args.max_queue_length,
        'time_estimator_error': args.time_estimator_error,
        'jobgen_spec': {
            'dataset': args.dataset,
            'arrival_rate': args.arrival_rate,
            'cv_factor': args.cv_factor,
            'total_jobs': args.total_jobs,
            'job_runtime': args.job_runtime,
            'seed': args.seed
        },
        # RL config
        'use_rl': args.use_rl,
        'train_rl': args.train_rl,
        'rl_workers': args.rl_workers,
        'rl_model_path': args.rl_model_path,
        'rl_save_path': args.rl_save_path,
        'rl_lr': args.rl_lr,
        'rl_gamma': args.rl_gamma,
        'rl_entropy_coeff': args.rl_entropy_coeff,
        'rl_value_coeff': args.rl_value_coeff,
        'rl_max_grad_norm': args.rl_max_grad_norm,
        'cloud_cost_sensitivity': args.cloud_cost_sensitivity,
        'cost_optimization_mode': args.cost_optimization_mode,
        'adaptive_scheduling': args.adaptive_scheduling,
        'multi_objective_weights': [float(x) for x in args.multi_objective_weights.split(',')],
        'enhanced_waiting_policy': args.enhanced_waiting_policy,
    }
    grid_search_config = utils.convert_to_lists(grid_search_config)
    run_configs = utils.generate_cartesian_product(grid_search_config)
    run_configs = apply_filter_config(args.filter_name, run_configs)

    final_simulator_results = run_grid_search(run_configs, num_procs=args.processes)
    if args.log:
        file_path = args.log
    else:
        file_path = None
    if args.log:
        absolute_file_path = os.path.abspath(file_path)
        dir_path = os.path.dirname(absolute_file_path)
        os.system(f'mkdir -p {dir_path}')
        file = open(absolute_file_path, 'wb')
        pickle.dump(final_simulator_results, file)
        file.close()
