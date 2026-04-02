import copy
from collections import deque

import numpy as np
from typing import Any, Dict, List, Optional

import torch
from tabulate import tabulate
from tqdm import tqdm

from skyburst import Cluster, Job, utils, waiting_policy
# from skyburst.waiting_policy import cardinal_query_wait

DEFAULT_SIMULATOR_SPEC = {
    # Size of the cluster (i.e. # of cluster nodes).
    'cluster_size': 64,
    # Number of GPU(s) per cluster node.
    'gpus_per_node': 8,
    # Number of CPU(s) per cluster node.
    'cpus_per_node': 96,
    # Scheduling algorithm specifying order of the queue.
    'sched_alg': 'fifo',
    # How jobs are binpacked into the cluster.
    'binpack_alg': 'first-fit',
    # Waiting policy (how long jobs should wait in the cloud).
    'waiting_policy': 'linear_runtime',
    # Waiting hyperparameter (to be passed to waiting_policy)
    'waiting_factor': 1.25,
    # Sets clipping time for waiting (max time a job should wait)
    'clip_time': 1e9,
    # Enable backfill (assumes time estimator).
    'backfill': False,
    # Enable loop scheduling (just loop through entire queue, remove HoL).
    'loop': False,
    # Enable prediction. (Jobs predict if they can be assigned to cluster before timing out).
    # 0 is no prediction, 1 is perfect oracle
    'predict_wait': 0,
    # Queue length
    'max_queue_length': -1,
    # Long jobs wait
    'long_job_thres': -1,
    # Time estimator error
    'time_estimator_error': 0,
    # Data locality dealy
    'data_gravity': -1,
    # Pre-empt Cloud Ratio.
    # 2 means that the waiting policy is 2 * (waiting time), where waiting time is determined by waiting policy. If jobs times out with this waitin gpolicy is it moved to the cloud.
    # If the job exceeds long_job_thres time on the cloud, it is moved back to the onprem.
    # The waiting time for onprem is now 2 * (waiting time) (for long jobs).
    'preempt_cloud_ratio': -1,
    # (Deprecated) Algorithm to immediately send job to cloud (without waiting).
    'filter_alg': None,
    # Prints out simulator state at every timestep.
    'verbose': False,
    # Appends python debugger at every timestemp.
    'debug': False,
    # Position for TQDM progress tracker bar.
    'pbar_idx': 0,
    # Jobs to not consider for final metrics at the beg. and end. of simulator.
    'warmup_jobs': 5000,
    # Whether to get snapshots and save to result dict
    'snapshot': False,
    'snapshot_interval': -1, # If > 0, takes periodic snapshots of node state
    # Metadata on job generation (run prior to simulator).
    'jobgen_spec': {
        # Dataset type ['philly', 'philly_gen', 'gen_gpu']
        'dataset': 'philly',
        # Arrival rate of jobs (used in 'gen_gpu', 'philly_gen')
        'arrival_rate': -1,
        # Total number of jobs generated.
        'total_jobs': -1,
        # Avg. Job runtime (used in 'gen_gpu')
        'job_runtime': -1,
    },
    # 强化学习配置
    'use_rl': False,
    'rl_model_path': None,
    'train_rl': False,
    'rl_workers': 4,
    'rl_lr': 0.0001,
    'rl_gamma': 0.99,
    'rl_save_path': 'models/a3c_scheduler.pth',
    # 基数查询参数
    'cardinal_rounds': 3,  # 基数查询轮数
    'adaptation_rate': 0.2,  # 权重调整速率
    'enable_dynamic_adjustment': True,  # 是否启用动态调整
    'stable_matching': True,  # 是否生成稳定匹配
}


def run_simulator(
        jobs: List[Job],
        simulator_spec: Optional[Dict[str, Any]] = DEFAULT_SIMULATOR_SPEC):
    """
    该模拟器通过离散事件仿真，为分布式资源管理系统提供了可配置的测试平台，
    能够帮助开发者优化调度算法和资源管理策略。
    Optional	                表示参数可以是后面类型或None	    类似"可空类型"概念
    Dict[str, Any]	            字典类型：键为字符串，值为任意类型	类似JSON对象
    = DEFAULT_SIMULATOR_SPEC	设置默认参数值	                当不传参时自动使用预定义配置
    """
    """Executes a simulator over a fixed set of jobs. Returns a result dictionary over all finished jobs.

    Args:
        # 按到达时间，排序生成的list
        jobs: List of generated jobs sorted by their arrival times.
        # 模拟器设置
        simulator_spec: Simulator settings, see above dictionary for default
                        values.
    """
    """
    jobs data：
    [Job(idx=0, resources={'GPUs': 32, 'CPUs': 160}, arr=0.0, run = 336.00416666666666, deadline=0.0, start=None)
    , Job(idx=1, resources={'GPUs': 0, 'CPUs': 4}, arr=175.12194444444444, run = 336.0022222222222, deadline=0.0, start=None)
    simulator_spec:
    {'backfill': 0, 'binpack_alg': 'first-fit', 'clip_time': 1000000000.0, 
    'cluster_size': 64, 'cpus_per_node': 48, 'data_gravity': -1, 'debug': False, 
    'filter_alg': None, 'gpus_per_node': 8, 'jobgen_spec': {
    'arrival_rate': None, 'cv_factor': 1.0, 'dataset': 'helios', 
    'job_runtime': 4.0, 'seed': 2024, 'total_jobs': None}, 'long_job_thres': -1, 
    'loop': 0, 'max_queue_length': -1, 'pbar_idx': 0, 'predict_wait': 0, 
    'preempt_cloud_ratio': -1, 'sched_alg': 'fifo', 'snapshot': 0, 
    'time_estimator_error': 0, 'verbose': False, 'waiting_factor': 1.25, 
    'waiting_policy': 'linear_runtime', 'warmup_jobs': 5000, 'use_rl': False, 'rl_model_path': None, 'train_rl': False, 'rl_workers': 4, 'rl_lr': 0.001, 'rl_gamma': 0.99, 'rl_save_path': 'models/a3c_scheduler.pth'}
    """
    _simulator_spec = DEFAULT_SIMULATOR_SPEC.copy()
    if simulator_spec:
        _simulator_spec.update(simulator_spec)
    simulator_spec = _simulator_spec

    # TODO(mluo): convert into class fields instead of manual indexing.
    # 调度算法，如FIFO
    sched_alg = simulator_spec['sched_alg']
    sort_func = utils.generate_sorting_function(sched_alg)

    # 解析等待策略--waiting_policy：Waiting policy, can be infinite, constant, compute, star
    # zero-1、constant-0.454、linear_cost_filter_cpu-0.04、linear_capacity_filter_cpu-0.234
    waiting_policy_str = simulator_spec['waiting_policy'].split('-')
    """
    格式标准化：
    强制要求策略参数只能通过 单个连字符 附加在策略名称后（如 linear-1.25），避免开发者或用户使用复杂格式（如 policy-param1-param2）导致解析歧义。
    参数数量限制：
    明确策略最多只能接受 一个参数，例如：
        linear → 无参数（默认使用代码中预设的系数）
        linear-1.25 → 指定线性系数为 1.25
    防御性编程：
    防止因格式错误导致后续代码逻辑崩溃（如尝试访问 waiting_policy_str[2] 时引发 IndexError）。
    不符合要求-->抛出异常：AssertionError
    """
    assert len(waiting_policy_str) <= 2
    if len(waiting_policy_str) == 2:
        simulator_spec['waiting_policy'] = waiting_policy_str[0]
        simulator_spec['waiting_factor'] = float(waiting_policy_str[1])

    waiting_fn = waiting_policy.lookup_linear_function(
        simulator_spec['waiting_policy'],
        waiting_factor=int(simulator_spec['waiting_factor']))
    # output：<function lookup_linear_function.<locals>.<lambda> at 0x748fb7392200>
    # print(waiting_fn)
    # first-fit、best-fit、worst-fit...
    binpack_alg = simulator_spec['binpack_alg']
    # 是否采用backfill
    backfill = simulator_spec['backfill']
    loop = simulator_spec['loop']
    predict_wait = simulator_spec['predict_wait']
    # Sets clipping time for waiting (max time a job should wait)
    clip_time = simulator_spec['clip_time']
    filter_alg = simulator_spec['filter_alg']
    max_queue_length = simulator_spec['max_queue_length']
    time_estimator_error = simulator_spec['time_estimator_error'] / 100.0
    long_job_thres = simulator_spec['long_job_thres']
    preempt_cloud_ratio = simulator_spec['preempt_cloud_ratio']
    data_gravity = simulator_spec['data_gravity']
    if preempt_cloud_ratio > 0:
        assert long_job_thres > 0, 'Must set long_job_thres > 0 if preempt_cloud_ratio > 0'
    # 确保 backfill 和 loop 两个参数不能同时为 True
    assert not (
            backfill and loop
    ), f'Must only set one option to be True - backfill:{backfill}, loop:{loop} '

    verbose = simulator_spec['verbose']
    debug = simulator_spec['debug']
    snapshot = simulator_spec['snapshot']
    snapshot_interval = simulator_spec.get('snapshot_interval', -1)
    next_snapshot_time = 0.0
    
    # Initialize simulator variables
    jobs = copy.deepcopy(jobs)
    queue = []
    finished_jobs = []
    num_jobs = len(jobs)
    cloud_cost = 0.0
    # Create fake cluster. The cluster is homogeneous.
    """
    Cluster State:
    Node 0: GPU: {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 
    6: None, 7: None}, CPU: 0
    Node 1: GPU: {0: ...None, 6: None, 7: None}, CPU: 0
    Node 63: GPU: {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 
    6: None, 7: None}, CPU: 0
    """
    # Create cluster instance
    cluster = Cluster(num_nodes=simulator_spec['cluster_size'],
                      num_gpus_per_node=simulator_spec['gpus_per_node'],
                      num_cpus_per_node=simulator_spec['cpus_per_node'],
                      backfill=backfill,
                      binpack=binpack_alg)
    
    # Configure DiscoRL Scheduler
    waiting_policy.configure_scheduler(simulator_spec)

    # 假设模拟器的当前时间
    t = 0
    pbar = tqdm(total=len(jobs),
                desc="Jobs progress: ",
                position=simulator_spec['pbar_idx'])

    snapshots = {}
    total_cloud_jobs = 0
    # Simulation Loop - Continues until all jobs have passed and the queue is empty and the cluster has no more jobs.
    while len(jobs) > 0 or len(queue) > 0 or cluster.active_jobs:
        # Clear cluster of jobs that have completed
        completed_jobs = cluster.try_clear(t)
        finished_jobs.extend(completed_jobs)

        # Check for jobs that have waited too long (move to cloud).--检查等待超时并发送到云端
        i = 0
        while i < len(queue):
            job = queue[i]
            # If job has timed out, send to cloud.
            if t > job.deadline - job.runtime:
                raise ValueError(
                    f'Job {job.idx} has timed out: {t} > {job.deadline}')
            elif t == job.deadline - job.runtime:
                """
                作业超时：更新作业状态，计算开始时间和云端成本，并将其添加到已完成作业列表。
                """
                queue.remove(job)
                # 标记了timeout-cloud-->直接在作业到达时-->发送到cloud
                job.state = 'TIMEOUT-CLOUD'
                # Shortcut: Job can predict it will go to cloud or not, if so, it would have began running at job.arrival.
                # Perfect Oracle
                if predict_wait == 1:
                    job.start = job.arrival
                elif predict_wait == 0 or predict_wait == 2:
                    job.start = job.deadline - job.runtime
                else:
                    raise ValueError(
                        f'Predict wait {predict_wait} wrong value!')
                # 云抢占机制
                if preempt_cloud_ratio > 0:
                    # If a job has not been prempted to the cloud before.
                    if not job.preempt_cloud:
                        # Emulate preemption from cloud back to onprem.
                        if job.runtime > long_job_thres:
                            # Job will run on the cloud for long_job_thres and then move back to onprem.
                            job.preempt_cloud = True
                            # The job will arrive again at this time (original arrival + original waiting time + long_job_thres)
                            job.new_arrival = job.deadline - job.runtime + long_job_thres
                            job.start = None
                            job.deadline = None
                            job.state = None

                            # Here, we add it back into the arrival jobs, sorted by arrival time.
                            # Finding the correct position for the new object
                            position = 0
                            for obj in jobs:
                                # Design Choice: Should we insert by its original arrival or new arrival?
                                # Here, we insert by the new arrival.
                                if obj.arrival > job.new_arrival:
                                    break
                                position += 1
                            # Inserting the object at the found position
                            jobs.insert(position, job)
                            # Cloud cost incurred includes the time job ran for LONG_JOB_THRES on the cloud.
                            # cloud cost==云消费+作业消费/（作业运行时间/长作业阈值）
                            cloud_cost += job.cost / (job.runtime / long_job_thres)
                            pbar.update(-1)
                            continue
                cloud_cost += job.cost
                total_cloud_jobs += 1
                # 添加到已完成作业列表
                finished_jobs.append(job)
            else:
                i += 1

        # Add jobs to queue that have arrived. Jobs are assumed to have been ordered by arrival times.
        i = 0
        while i < len(jobs):
            job = jobs[i]
            if job.arrival < t and job.new_arrival < t:
                raise ValueError("Should not have entered here!")
            elif job.arrival == t or job.new_arrival == t:
                jobs.remove(job)
                deadline = waiting_fn(job, cluster=cluster, cur_timestamp=t)
                if job.preempt_cloud:
                    arrival = job.new_arrival
                else:
                    arrival = job.arrival

                if preempt_cloud_ratio > 0:
                    # Mutate waiting function.==剩余时间
                    waiting_time = max(0.0, deadline - job.runtime - arrival)
                    if job.preempt_cloud:
                        # Star-Wait applies a longer waiting policy for cloud preempted jobs.
                        waiting_time = preempt_cloud_ratio * waiting_time
                    else:
                        # Star-Wait originally applies No-Wait policy
                        waiting_time = 0
                    deadline = arrival + job.runtime + waiting_time
                job.set_deadline(deadline)

                if deadline == -1 or (preempt_cloud_ratio < 0 and job.runtime < long_job_thres):
                    # For Constant-Wait + No-SJ (job offloading)
                    if hasattr(job, 'state'):
                        job.state = 'TIMEOUT-CLOUD'
                    if hasattr(job, 'start'):
                        job.start = arrival
                    job.set_deadline(deadline=arrival + job.runtime)
                    cloud_cost += job.cost
                    finished_jobs.append(job)
                else:
                    # For time estimator ablations.
                    if time_estimator_error != 0:
                        original_runtime = job.runtime
                        mod_runtime = original_runtime + np.random.normal(
                            loc=0.0,
                            scale=time_estimator_error * original_runtime)
                        mod_runtime = max(0, mod_runtime)
                        if 'filter' in simulator_spec['waiting_policy']:
                            deadline = arrival + simulator_spec[
                                'waiting_factor'] * (
                                               job.resources['GPUs'] +
                                               job.resources['CPUs'] /
                                               53.0) * mod_runtime + job.runtime
                    waiting_time = max(0.0,
                                       deadline - job.runtime - arrival)
                    if waiting_time < 0:
                        raise ValueError('Waiting time should not be negative.')
                    waiting_time = min(clip_time, waiting_time)
                    job.set_deadline(deadline=arrival + job.runtime +
                                              waiting_time)
                    queue.append(job)
                pbar.update(1)
            else:
                break

        # 对队列进行排序
        queue.sort(key=sort_func)

        # 使用传统方法处理剩余作业
        i = 0
        while i < len(queue):
            job = queue[i]
            can_fit, _ = cluster.try_fit_v2(t, job)
            if can_fit:
                queue.remove(job)
            elif not loop:
                break
            else:
                i += 1

        # Perform EASY backfilling (原有代码)
        if backfill:
            # Reserve the first element of queue that is blocking
            if queue:
                job_to_reserve = queue[0]
                # Reserving large jobs for backfilling
                can_reserve = cluster.try_reserve(t, job_to_reserve)
                # If can't reserve within reasonable time, leave the job in the queue.
                if not can_reserve:
                    pass
                else:
                    queue.remove(job_to_reserve)
                i = 0
                while i < len(queue):
                    job = queue[i]
                    can_fit, preempted_jobs = cluster.try_fit_v2(t, job)
                    if can_fit:
                        queue.remove(job)
                    else:
                        i += 1

        if max_queue_length != -1:
            while len(queue) > max_queue_length:
                q_job = queue[-1]
                queue.remove(q_job)
                q_job.state = 'TIMEOUT-CLOUD'
                q_job.start = t
                if q_job.preempt_cloud:
                    q_job.set_deadline(deadline=q_job.new_arrival +
                                                q_job.runtime)
                else:
                    q_job.set_deadline(deadline=q_job.arrival + q_job.runtime)
                cloud_cost += q_job.cost
                finished_jobs.append(q_job)

        if snapshot:
            # Standard Queue Snapshot (Backward Compatibility)
            if snapshot_interval <= 0:
                if t not in snapshots:
                    snapshots[t] = {}
                snapshots[t]['new_queue'] = copy.deepcopy(queue)
            
            # Periodic Node Snapshot (For Experiment 3)
            elif t >= next_snapshot_time:
                node_states = []
                for i, node in enumerate(cluster.nodes):
                    # Find jobs running on this node
                    node_jobs = []
                    for job in cluster.active_jobs.values():
                        # job.allocated_gpus is Dict[node_idx, List[gpu_idx]]
                        if i in job.allocated_gpus:
                            node_jobs.append(job)
                    
                    if not node_jobs:
                        node_states.append({'node_id': i, 'compute_util': 0.0, 'mem_util': 0.0})
                        continue
                        
                    total_gpu_util = sum(getattr(j, 'gpu_util', 0) * len(j.allocated_gpus[i]) for j in node_jobs)
                    total_mem_util = sum(getattr(j, 'mem_util', 0) * len(j.allocated_gpus[i]) for j in node_jobs)
                    
                    # Normalize: Util per GPU * Num GPUs Used / Total GPUs
                    # Assuming gpu_util is 0-100 per card? Or per job?
                    # Previous assumption: Job has gpu_util 0-100.
                    # If job uses 1 GPU, it contributes gpu_util.
                    # If job uses 2 GPUs, does it contribute 2*gpu_util? 
                    # Usually util is "average per card".
                    # Let's assume gpu_util is "utilization percentage of the allocated resource".
                    # So if allocated 1 GPU, load is gpu_util.
                    # Node Load = Sum(Job Util) / Node Capacity
                    
                    # Correction: In Exp 3, we injected 100.0 / 20.0.
                    # This means "100% of the GPU it asked for".
                    # So sum is correct.
                    
                    norm_compute = min(100, total_gpu_util / simulator_spec['gpus_per_node'])
                    norm_mem = min(100, total_mem_util / simulator_spec['gpus_per_node'])
                    
                    node_states.append({
                        'node_id': i,
                        'compute_util': norm_compute,
                        'mem_util': norm_mem
                    })
                
                if t not in snapshots:
                    snapshots[t] = {}
                snapshots[t]['nodes'] = node_states
                next_snapshot_time += snapshot_interval

        # Skip to next timestep (matches algorithm 1 in paper). The next timestep is the minimum of:
        # 1) a new job either arrives (first element in job queue)
        # 2) job finishes on the cluster
        # 3) existing job in the queue times out.

        # Case 2
        next_time_list = []
        for _, job in cluster.active_jobs.items():
            next_time_list.append(job.start + job.runtime)

        # Case 1
        if len(jobs) > 0:
            cur_job = jobs[0]
            if cur_job.preempt_cloud:
                next_time_list.append(cur_job.new_arrival)
            else:
                next_time_list.append(cur_job.arrival)

        # Case 3
        if queue:
            for q in queue:
                # append time outs
                next_time_list.append(q.deadline - q.runtime)
        
        # Case 4: Next Snapshot
        if snapshot and snapshot_interval > 0:
            next_time_list.append(next_snapshot_time)

        # If there are no jobs left in the cluster and in the job and queue, terminate simulation.
        if len(next_time_list) == 0:
            assert len(queue) == 0 and len(jobs) == 0
            break

        if min(next_time_list) < t and abs(min(next_time_list) - t) > 1e-6:
            print(simulator_spec)
            raise ValueError(
                f'Simulator cannot go back in time, there is a bug: {t}->{min(next_time_list)}'
            )
        t = min(next_time_list)
        if verbose or debug:
            headers = [
                'Timestamp', 'Cloud Cost', 'Queue Length', 'Jobs Left',
                'Finished Jobs'
            ]
            data = [(t, cloud_cost, len(queue), len(jobs), len(finished_jobs))]
            print(tabulate(data, headers=headers))
            if debug:
                import pdb
                pdb.set_trace()
        # 解析基数查询相关参数
        cardinal_rounds = simulator_spec.get('cardinal_rounds', 3)
        adaptation_rate = simulator_spec.get('adaptation_rate', 0.2)
        enable_dynamic_adjustment = simulator_spec.get('enable_dynamic_adjustment', False)
        stable_matching = simulator_spec.get('stable_matching', False)

        # 根据负载波动性动态调整适应率
        if hasattr(cluster, 'load_history') and len(cluster.load_history) > 10:
            load_variance = np.var(list(cluster.load_history))
            # 波动大时降低适应率，提高稳定性
            if load_variance > 0.1:
                adaptation_rate = max(0.05, adaptation_rate * 0.5)


    end_sim_jobs = cluster.try_clear(1e12)
    assert len(end_sim_jobs) == 0 and len(jobs) == 0 and len(
        queue) == 0, 'Simulator did not finish properly. There are still running jobs in the cluster.'

    # Sort jobs by their initial arrival (aka idx).
    finished_jobs.sort(key=lambda x: x.idx)

    # Generate final logs for the simulator.
    result_dict = {
        'idx': np.array([j.idx for j in finished_jobs]),
        'arrival': np.array([j.arrival for j in finished_jobs]),
        'start': np.array([j.start for j in finished_jobs]),
        'runtime': np.array([j.runtime for j in finished_jobs]),
        'deadline': np.array([j.deadline for j in finished_jobs]),
        'num_gpus': np.array([j.num_gpus for j in finished_jobs]),
        'state': np.array([j.state for j in finished_jobs]),
        'allocated_gpus': np.array([j.allocated_gpus for j in finished_jobs]),
        'simulator_spec': simulator_spec,
        'stats': {}
    }

    if snapshot:
        result_dict['snapshot'] = snapshots
    # Computing Simulator stats, such as avg. waiting, avg. JCT, cloud cost, utilization.
    total_waiting_time = 0.0
    total_running_time = 0.0
    num_jobs = 0
    total_cloud_cost = 0
    sum_local_space = 0.0
    sum_cloud_space = 0.0

    start_time = finished_jobs[simulator_spec['warmup_jobs']].arrival
    end_time = finished_jobs[len(finished_jobs) -
                             simulator_spec['warmup_jobs'] - 1].arrival

    jct_list = []
    wait_list = []
    if data_gravity != -1:
        for job in finished_jobs:
            if job.state == 'TIMEOUT-CLOUD':
                job.cost = job.cost * (1 + data_gravity / job.runtime)
                job.runtime = job.runtime + data_gravity
    for job in finished_jobs:
        inter_start = max(job.start, start_time)
        inter_end = min(job.start + job.runtime, end_time)
        # Cut off beginning and ending of simulator to reach steady state. Calculate the "bleeding".
        if job.idx < simulator_spec['warmup_jobs'] or job.idx > len(
                finished_jobs) - simulator_spec['warmup_jobs']:
            if job.state == 'LOCAL':
                if inter_end >= inter_start:
                    # to local
                    sum_local_space += job.num_gpus * (inter_end - inter_start)
            elif job.state == 'TIMEOUT-CLOUD':
                if inter_end >= inter_start:
                    sum_cloud_space += job.num_gpus * (inter_end - inter_start)
            continue
        # Moved to cloud
        if job.state == 'TIMEOUT-CLOUD':
            total_waiting_time += job.start - job.arrival
            if inter_end >= inter_start:
                sum_cloud_space += job.num_gpus * (inter_end - inter_start)
            total_cloud_cost += job.cost
        elif job.state == 'LOCAL':
            total_waiting_time += job.start - job.arrival
            if inter_end >= inter_start:
                sum_local_space += job.num_gpus * (inter_end - inter_start)
            if job.preempt_cloud:
                total_cloud_cost += job.cost / (job.runtime / long_job_thres)
        else:
            raise ValueError(f'Job {job.idx} has invalid state: {job.state}')
        jct_list.append(job.runtime + job.start - job.arrival)
        wait_list.append(job.start - job.arrival)
        total_running_time += job.runtime
        num_jobs += 1

    result_dict['stats']['total_cloud_cost'] = total_cloud_cost
    result_dict['stats']['avg_cloud_cost'] = total_cloud_cost / (end_time -
                                                                 start_time)
    result_dict['stats']['avg_waiting'] = total_waiting_time / num_jobs
    result_dict['stats']['avg_jct'] = (total_waiting_time +
                                       total_running_time) / num_jobs
    result_dict['stats']['90_jct'] = np.percentile(jct_list,
                                                   90,
                                                   method='nearest')
    result_dict['stats']['99_jct'] = np.percentile(jct_list,
                                                   99,
                                                   method='nearest')

    result_dict['stats']['avg_wait'] = np.mean(wait_list)
    result_dict['stats']['90_wait'] = np.percentile(wait_list,
                                                    90,
                                                    method='nearest')
    result_dict['stats']['99_wait'] = min(
        24.0, float(np.percentile(wait_list, 99, method='nearest')))

    result_dict['stats']['cluster_utilization'] = sum_local_space / (
            simulator_spec['cluster_size'] * simulator_spec['gpus_per_node'] *
            (end_time - start_time))
    result_dict['stats']['system_utilization'] = (
                                                         sum_local_space + sum_cloud_space) / (
                                                         simulator_spec['cluster_size'] *
                                                         simulator_spec['gpus_per_node'] *
                                                         (end_time - start_time))

    stats_dict = result_dict['stats']
    headers = [
        'Sched Policy', 'Waiting Policy', '# Cluster Nodes',
        'Total Cloud Cost', 'Avg. Cloud Cost', 'Avg. Waiting', 'Avg. JCT',
        '90th JCT', '99th JCT', 'Cluster Utilization', 'System Utilization'
    ]
    waiting_policy_str = simulator_spec['waiting_policy']
    waiting_factor_str = simulator_spec['waiting_factor']
    data = [(simulator_spec['sched_alg'], \
             f'{waiting_policy_str}-{waiting_factor_str}', simulator_spec['cluster_size'], \
             stats_dict['total_cloud_cost'], stats_dict['avg_cloud_cost'], \
             stats_dict['avg_waiting'], stats_dict['avg_jct'], stats_dict['90_jct'], stats_dict['99_jct'],
             stats_dict['cluster_utilization'], stats_dict['system_utilization'])]
    print(tabulate(data, headers=headers))

    # Save trained model if in training mode
    if simulator_spec['train_rl'] and simulator_spec.get('rl_save_path'):
        print(f"Saving DiscoRLScheduler model to {simulator_spec['rl_save_path']}")
        waiting_policy.save_scheduler_model(simulator_spec['rl_save_path'])

    return result_dict


# 计算当前系统负载
def _estimate_current_system_load(cluster, queue):
    """估计当前系统负载，考虑队列大小和集群利用率"""
    # 计算基础利用率
    base_utilization = 0.0
    if hasattr(cluster, 'nodes') and len(cluster.nodes) > 0:
        total_gpus = cluster.num_nodes * cluster.num_gpus_per_node
        used_gpus = sum([node.num_gpus_per_node - node.free_gpus for node in cluster.nodes])
        total_cpus = cluster.num_nodes * cluster.num_cpus_per_node
        used_cpus = sum([node.num_cpus_per_node - node.free_cpus for node in cluster.nodes])

        gpu_util = used_gpus / max(1, total_gpus)
        cpu_util = used_cpus / max(1, total_cpus)
        base_utilization = max(gpu_util, cpu_util)

    # 考虑队列长度因素
    queue_factor = len(queue) / max(10, cluster.num_nodes * 0.5)
    queue_factor = min(2.0, queue_factor)  # 限制最大影响

    # 综合负载估计
    system_load = max(base_utilization, 0.5 * queue_factor)
    system_load = min(3.0, max(0.5, system_load))  # 范围限制

    return system_load
