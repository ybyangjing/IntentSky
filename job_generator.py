import copy
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from skyburst import Job, waiting_policy
from skyburst.traces import philly
from skyburst.traces import helios
from skyburst.traces import dag

# Returns the total cost of a GPU-only job.
def gpu_cost_fn(resources: dict, runtime: float):
    return resources['GPUs'] * runtime


# Returns the total cost of a GPU/CPU hybrid job.
def hybrid_cost_fn(resources: dict, runtime: float):
    return 50 * resources['GPUs'] * runtime + resources['CPUs'] * runtime


def load_processed_jobs(dataset_config: Dict[str, Any]):
    dataset_type = dataset_config['dataset']
    if dataset_type == 'philly':
        philly_jobs = philly.load_philly_traces('~/philly-traces/trace-data')
        return process_philly_jobs(philly_jobs)
    elif dataset_type == 'philly_dag':
        # Use the caching DAG loader
        return dag.load_dag_traces_with_cache(dataset_config)
    elif dataset_type == 'philly_gen':
        philly_jobs = philly.load_philly_traces('~/philly-traces/trace-data')
        dataset_kwargs = {
            'total_jobs': dataset_config['total_jobs'],
            'arrival_rate': dataset_config['arrival_rate'],
            'cv_factor': dataset_config['cv_factor'],
            'seed': dataset_config['seed'],
        }
        return generate_philly_gpu_jobs(philly_jobs, **dataset_kwargs)
    elif dataset_type == 'gen_gpu':
        philly_jobs = philly.load_philly_traces('~/philly-traces/trace-data')
        dataset_kwargs = {
            'total_jobs': dataset_config['total_jobs'],
            'arrival_rate': dataset_config['arrival_rate'],
            'job_runtime': dataset_config['job_runtime'],
            'seed': dataset_config['seed'],
        }
        return generate_gpu_jobs(philly_jobs, **dataset_kwargs)
    elif dataset_type == 'helios':
        helios_jobs = helios.load_helios_traces('~/HeliosData/data/Venus')
        return process_helios_jobs(helios_jobs)
    elif dataset_type == 'helios_dag':
        # Use the caching DAG loader
        return dag.load_dag_traces_with_cache(dataset_config)
    elif dataset_type == 'helios_blocked':
        # Experiment 3: Blocked Workload for Fragmentation Analysis
        return generate_blocked_helios_jobs(dataset_config)
    elif dataset_type == 'philly_blocked':
        # Experiment 3: Blocked Workload from Philly Traces
        return generate_blocked_philly_jobs(dataset_config)
    elif dataset_type == 'philly_privacy':
        # Experiment 5: Privacy & Data Gravity Workload
        return generate_privacy_philly_jobs(dataset_config)
    elif dataset_type == 'helios_gen':
        helios_jobs = helios.load_helios_traces('~/HeliosData/data/Venus')
        dataset_kwargs = {
            'total_jobs': dataset_config['total_jobs'],
            'arrival_rate': dataset_config['arrival_rate'],
            'cv_factor': dataset_config['cv_factor'],
            'seed': dataset_config['seed'],
        }
        return generate_helios_jobs(helios_jobs, **dataset_kwargs)
    elif dataset_type == 'synthetic':
        dataset_kwargs = {
            'total_jobs': dataset_config['total_jobs'],
            'arrival_rate': dataset_config['arrival_rate'],
            'job_runtime': dataset_config['job_runtime'],
            'cv_factor': dataset_config['cv_factor'],
            'seed': dataset_config['seed'],
        }
        return generate_synthetic_jobs(**dataset_kwargs)
    else:
        raise ValueError(
            f'Dataset {dataset_type} does not exist or has not been implemented yet.'
        )


def process_philly_jobs(philly_jobs: List['JobTrace']):
    """Converts entire Philly job trace into a list of simulator jobs.
    """
    jobs = philly_jobs.copy()
    # Remove invalid jobs (jobs that have not finished and jobs that failed/killed early)
    jobs = [j for j in jobs if j._run_time is not None and j.status == 'Pass']
    jobs.sort(key=lambda j: j._submitted_time)

    # Arrival time for jobs
    start_time = jobs[0]._submitted_time
    arrival_times = [(j._submitted_time - start_time).total_seconds() / 3600.0
                     for j in jobs]

    # Run time for jobs
    run_times = [j._run_time / 60.0 for j in jobs]

    # Get GPU resources
    resources = []
    for j in jobs:
        gpu_count = sum(
            [len(node_dict['gpus']) for node_dict in j.attempts[-1]['detail']])
        resources.append({'GPUs': gpu_count})

    costs = [res['GPUs'] * run for res, run in zip(resources, run_times)]

    return [Job(idx, arrival=arr, runtime=run, resources=res, cost=cost) \
            for idx, (arr, run, res, cost) in \
            enumerate(list(zip(arrival_times, run_times, resources, costs)))]


def generate_philly_gpu_jobs(philly_jobs: List['JobTrace'],
                             arrival_rate=32.0,
                             cv_factor=1.0,
                             total_jobs=300000,
                             seed=2024):
    """Generates Philly jobs based on a Poisson arrival distribution.

    Interarrival times follow an exponential distribution of 1/arrival_rate.
    Jobs are randomly sampled from the Philly job trace.
    """
    total_jobs = int(total_jobs)
    jobs = philly_jobs.copy()
    # Remove invalid jobs (jobs that have not finished and jobs that failed/killed early)
    jobs = [j for j in jobs if j._run_time is not None and j.status == 'Pass']
    jobs.sort(key=lambda j: j._submitted_time)

    # Arrival time for jobs
    np.random.seed(seed)
    # Check if cv_factor is None or invalid, default to 1.0 if so
    if cv_factor is None or cv_factor <= 0:
        cv_factor = 1.0
        
    alpha = (1.0 / cv_factor)**2
    # Check if arrival_rate is None or invalid
    if arrival_rate is None or arrival_rate <= 0:
        # Default arrival rate if not provided? Or raise error?
        # Assuming 32.0 as default based on function signature, but if None passed, use it.
        arrival_rate = 32.0

    interarrival_times = np.array([
        np.random.gamma(shape=alpha, scale=1 / (alpha * arrival_rate))
        for _ in range(total_jobs - 1)
    ])
    # interarrival_times = np.random.exponential(scale=1 / arrival_rate,
    #                                            size=total_jobs - 1)
    interarrival_times = np.insert(interarrival_times, 0, 0)
    arrival_times = np.cumsum(interarrival_times)

    # Run time for jobs
    run_times = []
    for j in jobs:
        run_time_hr = j._run_time / 60.0
        run_times.append(run_time_hr)

    # Get GPU resources
    resources = []
    for j in jobs:
        detail_dict = j.attempts[-1]['detail']
        gpu_count = sum([len(node_dict['gpus']) for node_dict in detail_dict])
        resources.append({'GPUs': gpu_count})
    np.random.seed(seed)
    job_indexes = np.random.choice(list(range(len(run_times))),
                                   size=total_jobs,
                                   replace=True)
    proc_jobs = []
    for idx in range(total_jobs):
        job_idx = job_indexes[idx]
        resources_dict = resources[job_idx]
        runtime = run_times[job_idx]
        cost = resources_dict['GPUs'] * runtime
        proc_jobs.append(
            Job(idx,
                arrival=arrival_times[idx],
                runtime=runtime,
                resources=resources_dict,
                cost=cost))
    return proc_jobs


def generate_helios_jobs(helios_jobs: List['HeliosJobTrace'],
                         arrival_rate=32.0,
                         cv_factor=1.0,
                         total_jobs=300000,
                         seed=2024):
    """Converts entire Helios job trace into a list of simulator jobs.
    """
    jobs = helios_jobs.copy()
    # Remove invalid jobs (jobs that have not finished and jobs that failed/killed early)
    jobs = [
        j for j in jobs
        if j._run_time is not None and (j.status in ['COMPLETED', 'TIMEOUT'])
    ]
    jobs.sort(key=lambda j: j._submitted_time)

    # Arrival time for jobs
    np.random.seed(seed)
    # Check if cv_factor is None or invalid, default to 1.0 if so
    if cv_factor is None or cv_factor <= 0:
        cv_factor = 1.0
        
    alpha = (1.0 / cv_factor)**2
    
    # Check if arrival_rate is None or invalid
    if arrival_rate is None or arrival_rate <= 0:
        arrival_rate = 32.0
        
    interarrival_times = np.array([
        np.random.gamma(shape=alpha, scale=1 / (alpha * arrival_rate))
        for _ in range(total_jobs - 1)
    ])
    # interarrival_times = np.random.exponential(scale=1 / arrival_rate,
    #                                            size=total_jobs - 1)
    interarrival_times = np.insert(interarrival_times, 0, 0)
    arrival_times = np.cumsum(interarrival_times)

    # Run time for jobs
    run_times = [j._run_time for j in jobs]
    nodes = [j._nodes for j in jobs]

    # Get GPU resources
    resources = []
    for j in jobs:
        resources.append({'GPUs': j.num_gpus, 'CPUs': j.num_cpus})

    costs = [(res['GPUs'] + res['CPUs'] / 53.0) * run
             for res, run in zip(resources, run_times)]

    np.random.seed(seed)
    job_indexes = np.random.choice(list(range(len(run_times))),
                                   size=total_jobs,
                                   replace=True)
    proc_jobs = []
    for idx in range(total_jobs):
        job_idx = job_indexes[idx]
        resources_dict = resources[job_idx]
        runtime = run_times[job_idx]
        cost = costs[job_idx]
        num_nodes = nodes[job_idx]
        job = Job(idx,
                  arrival=arrival_times[idx],
                  runtime=runtime,
                  resources=resources_dict,
                  cost=cost)
        job.nodes = num_nodes  # 确保设置nodes属性
        proc_jobs.append(job)
    return proc_jobs


def generate_gpu_jobs(philly_jobs: List['JobTrace'],
                      arrival_rate=32.0,
                      job_runtime=4.0,
                      total_jobs=200000,
                      seed=2024):
    """Generates GPU jobs based on a Poisson arrival distribution and exponential runtime distribution.
    """
    total_jobs = int(total_jobs)
    jobs = philly_jobs.copy()
    # Remove invalid jobs (jobs that have not finished and jobs that failed/killed early)
    jobs = [j for j in jobs if j._run_time is not None and j.status == 'Pass']
    jobs.sort(key=lambda j: j._submitted_time)

    # Arrival time for jobs
    np.random.seed(seed)
    interarrival_times = np.random.exponential(scale=1 / arrival_rate,
                                               size=total_jobs - 1)
    interarrival_times = np.insert(interarrival_times, 0, 0)
    arrival_times = np.cumsum(interarrival_times)

    # Run time for jobs
    run_times = np.random.exponential(scale=job_runtime, size=total_jobs)

    # Get GPU resources
    resources = []
    for j in jobs:
        detail_dict = j.attempts[-1]['detail']
        gpu_count = sum([len(node_dict['gpus']) for node_dict in detail_dict])
        resources.append({'GPUs': gpu_count})
    np.random.seed(seed)
    job_indexes = np.random.choice(list(range(len(resources))),
                                   size=total_jobs,
                                   replace=True)
    proc_jobs = []
    for idx in range(total_jobs):
        job_idx = job_indexes[idx]
        runtime = run_times[idx]
        resources_dict = resources[job_idx]
        cost = resources_dict['GPUs'] * runtime
        proc_jobs.append(
            Job(idx,
                arrival=arrival_times[idx],
                runtime=runtime,
                resources=resources_dict,
                cost=cost))
    return proc_jobs


def process_helios_jobs(helios_jobs: List['HeliosJobTrace']):
    """Converts entire Helios job trace into a list of simulator jobs.
    """
    jobs = helios_jobs.copy()
    # Remove invalid jobs (jobs that have not finished and jobs that failed/killed early)
    jobs = [
        j for j in jobs
        if j._run_time is not None and (j.status in ['COMPLETED', 'TIMEOUT'])
    ]
    jobs.sort(key=lambda j: j._submitted_time)

    # Arrival time for jobs
    start_time = jobs[0]._submitted_time
    arrival_times = [(j._submitted_time - start_time).total_seconds() / 3600.0
                     for j in jobs]

    # Run time for jobs
    run_times = [j._run_time for j in jobs]
    nodes = [j._nodes for j in jobs]

    # Get GPU resources
    resources = []
    for j in jobs:
        resources.append({'GPUs': j.num_gpus, 'CPUs': j.num_cpus})

    costs = [(res['GPUs'] + res['CPUs'] / 53.0) * run
             for res, run in zip(resources, run_times)]

    processed_jobs = []
    for idx, (arr, run, res, cost, node) in enumerate(list(zip(arrival_times, run_times, resources, costs, nodes))):
        job = Job(idx, arrival=arr, runtime=run, resources=res, cost=cost)
        job.nodes = node  # 确保设置nodes属性
        processed_jobs.append(job)
    return processed_jobs


def generate_synthetic_jobs(arrival_rate=8.0,
                            job_runtime=1.0,
                            cv_factor=1.0,
                            total_jobs=20000,
                            seed=2025):
    """Generates GPU jobs based on a Poisson arrival distribution and exponential runtime distribution.
    """
    total_jobs = total_jobs

    # Arrival time for jobs
    np.random.seed(seed)
    alpha = (1.0 / cv_factor)**2
    interarrival_times = np.array([
        np.random.gamma(shape=alpha, scale=1 / (alpha * arrival_rate))
        for _ in range(total_jobs - 1)
    ])
    # interarrival_times = np.random.exponential(scale=1 / arrival_rate,
    #                                            size=total_jobs - 1)
    interarrival_times = np.insert(interarrival_times, 0, 0)
    arrival_times = np.cumsum(interarrival_times)

    # Run time for jobs
    run_times = np.random.exponential(scale=job_runtime, size=total_jobs)

    # Get GPU resources
    proc_jobs = []
    categorical = [0.7, 0.15, 0.1, 0.05]
    sizes = [1, 2, 4, 8]
    for idx in range(total_jobs):

        resources_dict = {'GPUs': np.random.choice(sizes, p=categorical)}
        temp = run_times[idx]
        cost = resources_dict['GPUs'] * temp
        proc_jobs.append(
            Job(idx,
                arrival=arrival_times[idx],
                runtime=temp,
                resources=resources_dict,
                cost=cost))
    return proc_jobs

def generate_blocked_helios_jobs(dataset_config):
    """
    Experiment 3 Special: Blocked Workload from Real Traces.
    Generates a workload where Generators (High Mem) arrive first, followed by Encoders (High Compute).
    """
    print(f"Generating Blocked Workload (Experiment 3) from helios_dag...")
    
    # 1. Load Real Data using standard DAG loader
    # Create a temporary config to load the base dataset
    base_config = dataset_config.copy()
    base_config['dataset'] = 'helios_dag'
    if base_config.get('total_jobs') is None or base_config.get('total_jobs') < 1000:
        base_config['total_jobs'] = 2000
        
    try:
        real_jobs = dag.load_dag_traces_with_cache(base_config)
    except Exception as e:
        print(f"Warning: Failed to load helios_dag ({e}). Returning empty list.")
        return []

    # 2. Filter Candidates
    # Heuristic: Sort by runtime to find candidates
    sorted_jobs = sorted(real_jobs, key=lambda x: x.runtime)
    
    # Take short jobs as Encoders (Compute Bound)
    encoders = sorted_jobs[:250]
    # Take long jobs as Generators (Memory Bound)
    generators = sorted_jobs[-250:]
    
    print(f"DEBUG: Encoder Count: {len(encoders)}, Avg Runtime: {np.mean([j.runtime for j in encoders])}")
    print(f"DEBUG: Generator Count: {len(generators)}, Avg Runtime: {np.mean([j.runtime for j in generators])}")
    
    experiment_jobs = []
    
    # Constants for resource capping (matching Experiment 3 constants)
    GPUS_PER_NODE = 8
    CPUS_PER_NODE = 96
    
    # 3. Construct Blocked Arrival & Inject Attributes
    # Block 1: Generators (High Mem) - Arrive early (t=0 onwards)
    for i, job in enumerate(generators):
        new_job = copy.deepcopy(job)
        new_job.arrival = i * 0.1
        
        # Inject Exp 3 Attributes
        new_job.gpu_util = 20.0
        new_job.mem_util = 100.0
        new_job.job_type = 'Generator'
        
        # Fix resources
        if new_job.resources.get('GPUs', 0) == 0:
             new_job.resources['GPUs'] = 1
        new_job.resources['GPUs'] = min(new_job.resources['GPUs'], GPUS_PER_NODE)
        
        if 'CPUs' in new_job.resources:
            new_job.resources['CPUs'] = min(new_job.resources['CPUs'], CPUS_PER_NODE)
             
        new_job.dependency_parent = None
        experiment_jobs.append(new_job)
        
    # Block 2: Encoders (High Compute) - Arrive later (t=10 onwards)
    for i, job in enumerate(encoders):
        new_job = copy.deepcopy(job)
        new_job.arrival = 10 + i * 0.1
        
        # Inject Exp 3 Attributes
        new_job.gpu_util = 100.0
        new_job.mem_util = 20.0
        new_job.job_type = 'Encoder'
        # Force runtime to be visible in snapshots
        new_job.runtime = max(new_job.runtime, 15.0)
        
        if new_job.resources.get('GPUs', 0) == 0:
             new_job.resources['GPUs'] = 1
        new_job.resources['GPUs'] = min(new_job.resources['GPUs'], GPUS_PER_NODE)
        
        if 'CPUs' in new_job.resources:
            new_job.resources['CPUs'] = min(new_job.resources['CPUs'], CPUS_PER_NODE)

        new_job.dependency_parent = None
        experiment_jobs.append(new_job)
    
    # Sort by arrival
    experiment_jobs.sort(key=lambda x: x.arrival)
    
    print(f"Generated {len(experiment_jobs)} blocked jobs for Experiment 3.")
    return experiment_jobs


def generate_blocked_philly_jobs(dataset_config):
    """
    Experiment 3 Special: Blocked Workload from Philly Traces.
    Generates a workload where Generators (High Mem) arrive first, followed by Encoders (High Compute).
    """
    print(f"Generating Blocked Workload (Experiment 3) from philly_dag...")
    
    # 1. Load Real Data using standard DAG loader
    # Create a temporary config to load the base dataset
    base_config = dataset_config.copy()
    base_config['dataset'] = 'philly_dag'
    if base_config.get('total_jobs') is None or base_config.get('total_jobs') < 1000:
        base_config['total_jobs'] = 2000
        
    try:
        real_jobs = dag.load_dag_traces_with_cache(base_config)
    except Exception as e:
        print(f"Warning: Failed to load philly_dag ({e}). Returning empty list.")
        return []

    # 2. Filter Candidates
    # Heuristic: Sort by runtime to find candidates
    sorted_jobs = sorted(real_jobs, key=lambda x: x.runtime)
    
    # Take short jobs as Encoders (Compute Bound)
    encoders = sorted_jobs[:250]
    # Take long jobs as Generators (Memory Bound)
    generators = sorted_jobs[-250:]
    
    experiment_jobs = []
    
    # Constants for resource capping (matching Experiment 3 constants)
    GPUS_PER_NODE = 8
    CPUS_PER_NODE = 96
    
    # 3. Construct Blocked Arrival & Inject Attributes
    # Block 1: Generators (High Mem) - Arrive early (t=0 onwards)
    for i, job in enumerate(generators):
        new_job = copy.deepcopy(job)
        new_job.arrival = i * 0.1
        
        # Inject Exp 3 Attributes (Optimized for 80% Target)
        new_job.gpu_util = 60.0  # Increased from 20 to 60
        new_job.mem_util = 100.0
        new_job.job_type = 'Generator'
        
        # Fix resources
        new_job.resources['GPUs'] = 1
        new_job.resources['CPUs'] = 1
        new_job.runtime = 20.0
             
        new_job.dependency_parent = None
        experiment_jobs.append(new_job)
        
    # Block 2: Encoders (High Compute) - Arrive later (t=10 onwards)
    for i, job in enumerate(encoders):
        new_job = copy.deepcopy(job)
        new_job.arrival = 5 + i * 0.1 # Arrive earlier (was 10) to mix better
        
        # Inject Exp 3 Attributes (Optimized for 80% Target)
        new_job.gpu_util = 100.0
        new_job.mem_util = 60.0  # Increased from 20 to 60
        new_job.job_type = 'Encoder'
        
        # Fix resources
        new_job.resources['GPUs'] = 1
        new_job.resources['CPUs'] = 1
        new_job.runtime = 20.0

        new_job.dependency_parent = None
        experiment_jobs.append(new_job)
    
    # Sort by arrival
    experiment_jobs.sort(key=lambda x: x.arrival)
    
    print(f"Generated {len(experiment_jobs)} blocked jobs for Experiment 3 (Philly).")
    return experiment_jobs

def generate_privacy_philly_jobs(dataset_config):
    """
    Experiment 5: Privacy Compliance & Data Gravity.
    Generates a workload with Sensitive (Privacy), Heavy (Gravity), and Normal jobs.
    """
    print(f"Generating Privacy & Gravity Workload (Experiment 5) from philly_dag...")
    
    # 1. Load Real Data
    base_config = dataset_config.copy()
    base_config['dataset'] = 'philly_dag'
    if base_config.get('total_jobs') is None:
        base_config['total_jobs'] = 1000
    
    try:
        real_jobs = dag.load_dag_traces_with_cache(base_config)
    except Exception as e:
        print(f"Warning: Failed to load philly_dag ({e}). Returning empty list.")
        return []
    
    # Slice to total_jobs to avoid processing entire dataset
    limit = dataset_config.get('total_jobs', 1000)
    if limit and len(real_jobs) > limit:
        real_jobs = real_jobs[:limit]
    
    # 2. Inject Attributes based on probabilities
    # 20% Sensitive (Must be on-prem)
    # 30% Heavy (Large Data Transfer)
    # 50% Normal
    
    experiment_jobs = []
    np.random.seed(2026) # Fixed seed for reproducibility
    
    # Sort by arrival first to keep order
    real_jobs.sort(key=lambda x: x.arrival)
    
    for i, job in enumerate(real_jobs):
        new_job = copy.deepcopy(job)
        
        # Compress arrival times to force contention (Burst Arrival)
        # Every 0.1 seconds a new job arrives
        new_job.arrival = i * 0.1
        
        # Random role assignment
        rand_val = np.random.random()
        
        if rand_val < 0.2:
            # Sensitive Job (Privacy Constraint)
            new_job.job_type = 'Sensitive'
            new_job.is_private = True
            new_job.is_sensitive = True
            new_job.data_size_gb = np.random.uniform(1, 5) # Small data, but sensitive
        elif rand_val < 0.5:
            # Heavy Job (Data Gravity)
            new_job.job_type = 'Heavy'
            new_job.is_private = False
            new_job.is_sensitive = False
            new_job.data_size_gb = np.random.uniform(50, 200) # 50-200 GB
            new_job.is_heavy = True
        else:
            # Normal Job
            new_job.job_type = 'Normal'
            new_job.is_private = False
            new_job.is_sensitive = False
            new_job.data_size_gb = np.random.uniform(1, 10) # 1-10 GB
            
        # Clear dependencies to simplify this specific experiment if needed, 
        # but maintaining DAGs is fine if we just analyze individual job placement.
        # For simplicity in "Privacy Violation" counting, treating them as individual jobs is easier.
        # But if we want HEFT to work, we might need dependencies. 
        # Let's keep dependencies but ensure attributes are propagated or set on nodes.
        
        experiment_jobs.append(new_job)
        
    print(f"Generated {len(experiment_jobs)} jobs for Experiment 5.")
    return experiment_jobs
