import itertools
import pickle
from typing import Any, List

import pandas as pd


def generate_sorting_function(sched_alg: str):
    """
    生成排序算法
    TODO：考虑加入PSRS排序算法 or 直接用RL解决
    https://github.com/xzli8/PSRS/blob/master/src/PSRS.c#L186
    """
    if sched_alg == 'fifo':
        # 使用任务的到达时间（arrival）进行排序，preempt_cloud 表示是否任务被云端抢占。
        # 如果没有抢占，使用 arrival 排序，如果抢占了，则使用 new_arrival 排序。
        # 修改为：短作业优先的fifo混合策略
        # def fifo_with_short_job_priority(job):
        #     arrival_time = job.arrival if not job.preempt_cloud else job.new_arrival
        #     # 短作业判断：运行时间小于0.5小时
        #     if job.runtime < 0.5:
        #         # 短作业组（优先级1）
        #         return (0, arrival_time)
        #     else:
        #         # 常规作业组（优先级2）
        #         return (1, arrival_time)

        # sort_func = fifo_with_short_job_priority
        sort_func = lambda x: x.arrival if not x.preempt_cloud else x.new_arrival
    # lifo（后进先出）
    elif sched_alg == 'lifo':
        sort_func = lambda x: -x.arrival if not x.preempt_cloud else x.new_arrival
    # edf（最早截止时间优先，Earliest Deadline First）
    elif sched_alg == 'edf':
        sort_func = lambda x: x.deadline
    elif sched_alg == 'evdf':
        # 按任务的加权截止时间排序，权重是任务需要的 GPU 数量（num_gpus）
        sort_func = lambda x: x.deadline * x.num_gpus
    elif sched_alg == 'ldf':
        sort_func = lambda x: -x.deadline
    elif sched_alg == 'sjf':
        sort_func = lambda x: x.runtime
    elif sched_alg == 'svjf':
        # 按任务的成本（cost）排序，成本最小的任务优先。
        sort_func = lambda x: x.cost
    elif sched_alg == 'ljf':
        sort_func = lambda x: -x.runtime
    elif sched_alg == 'lvjf':
        sort_func = lambda x: -x.cost
    elif sched_alg == 'swf':
        # 最短剩余时间优先:按任务的剩余时间排序，剩余时间最短的任务优先
        sort_func = lambda x: x.deadline - x.runtime
    elif sched_alg == 'svwf':
        # 按任务的剩余时间加权排序，权重是任务需要的 GPU 数量
        sort_func = lambda x: (x.deadline - x.runtime) * x.num_gpus
    elif sched_alg == 'lwf':
        sort_func = lambda x: -x.deadline + x.runtime
    else:
        raise ValueError(
            f'Scheudling algorithm {sched_alg} does not match existing policies.'
        )
    return sort_func


def convert_to_lists(d: dict):
    for key, value in d.items():
        # If the value is a dictionary, recursively convert it to a list
        if isinstance(value, dict):
            d[key] = convert_to_lists(value)
        elif not isinstance(value, list):
            d[key] = [value]
    return d


def flatten_dict(nested_dict, parent_key='', sep=':', preserve_name=False):
    flattened_dict = {}
    for key, value in nested_dict.items():
        if preserve_name:
            new_key = key
        else:
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flattened_dict.update(
                flatten_dict(value,
                             new_key,
                             sep=sep,
                             preserve_name=preserve_name))
        else:
            flattened_dict[new_key] = value
    return flattened_dict


def unflatten_dict(flattened_dict, sep=':'):
    unflattened_dict = {}
    for key, value in flattened_dict.items():
        parts = key.split(sep)
        current_dict = unflattened_dict
        for part in parts[:-1]:
            current_dict = current_dict.setdefault(part, {})
        current_dict[parts[-1]] = value
    return unflattened_dict


def generate_cartesian_product(d: dict):
    d = flatten_dict(d)
    print(d)
    # Get the keys and values from the outer dictionary
    keys = list(d.keys())
    values = list(d.values())

    # Use itertools.product to generate the cartesian product of the values
    product = itertools.product(*values)

    # Create a list of dictionaries with the key-value pairs for each combination
    result = [dict(zip(keys, p)) for p in product]
    # Return the list of dictionaries
    return [unflatten_dict(r) for r in result]


def is_subset(list1: List[Any], list2: List[Any]):
    """Checks if list2 is a subset of list1 and returns the matching indexes of the subset."""
    indexes = []
    for i2, elem in enumerate(list2):
        for i1, x in enumerate(list1):
            if x == elem and i1 not in indexes:
                indexes.append(i1)
                break
        if len(indexes) != i2 + 1:
            return []
    return indexes


def _load_logs(file_path: str):
    file = open(file_path, 'rb')
    return pickle.load(file)


def load_logs_as_dataframe(file_path: str):
    simulator_results = _load_logs(file_path)
    for r in simulator_results:
        if 'snapshot' in r:
            r['snapshot'] = [r['snapshot']]
    simulator_results = [
        flatten_dict(r, preserve_name=True) for r in simulator_results
    ]
    return pd.DataFrame(simulator_results)
