from skyburst.node import Node
from skyburst import utils

blocked_by_gpu_cpu_job = 0
blocked_by_cpu_job = 0


class Cluster(object):
    def __init__(self,
                 num_nodes,
                 num_gpus_per_node=8,
                 num_cpus_per_node=96,
                 binpack='first-fit',
                 backfill=False):
        self.num_nodes = num_nodes
        self.num_gpus_per_node = num_gpus_per_node
        self.num_cpus_per_node = num_cpus_per_node
        # List of nodes in the cluster. Assumed homogeneity.
        self.nodes = [
            Node(num_gpus_per_node, num_cpus_per_node)
            for _ in range(num_nodes)
        ]
        # Maps Job ID to Job, active jobs running in the cluster
        self.active_jobs = {}
        # Maps Job ID to Job, reserved jobs to be scheduled in cluster
        self.reserved_jobs = {}
        # This determines whether to binpack with backfill scheduling.
        self.backfill = backfill
        # Defines the bin packing algorithm, `first-fit`, `best-fit`.
        self.binpack = binpack

    def is_full(self):
        return all([n.free_gpus == 0 for n in self.nodes])

    def get_active_jobs(self):
        return self.active_jobs

    def try_fit_v2(self, cur_timestamp, job):
        global blocked_by_cpu_job
        global blocked_by_gpu_cpu_job
        num_gpus = job.resources['GPUs']
        num_cpus = job.resources['CPUs']
        num_cpus_per_node = num_cpus / job.nodes

        free_gpus = [n.free_gpus for n in self.nodes]
        free_cpus = [n.free_cpus for n in self.nodes]
        # Quick check, no hope of fitting onto cluster :(
        if num_gpus > sum(free_gpus) or num_cpus > sum(free_cpus):
            return False, []

        # Generate job GPU demands
        if job.nodes == 1:
            if num_gpus > self.num_gpus_per_node:
                # Assume worst case colocation
                # Multinode case, i.e. 26 GPUs, 8 GPU/node cluster -> job_gpu_demands = [8,8,8,2]
                job_gpu_demands = [self.num_gpus_per_node] * int(
                    num_gpus / self.num_gpus_per_node)
                if num_gpus % self.num_gpus_per_node:
                    job_gpu_demands.append(num_gpus % self.num_gpus_per_node)
            else:
                job_gpu_demands = [num_gpus]
        else:
            job_gpu_demands = [int(num_gpus / job.nodes)] * job.nodes

        # =============================================================================
        # Generate Job Plans
        # =============================================================================
        # Go through free space only first, generate partial plan with free space
        node_free_gpu_list = [
            list(range(self.num_gpus_per_node)) for _ in range(self.num_nodes)
        ]
        node_free_cpu_count = [self.num_cpus_per_node] * self.num_nodes

        # Go through active jobs
        for a_job_idx, a_job in self.active_jobs.items():
            a_job_cpu_per_node = a_job.num_cpus / a_job.nodes
            for n_idx, gpu_list in a_job.allocated_gpus.items():
                for gpu_idx in gpu_list:
                    node_free_gpu_list[n_idx].remove(gpu_idx)
                node_free_cpu_count[n_idx] -= a_job_cpu_per_node

        # Go through reserved jobs
        for r_job_idx, r_job in self.reserved_jobs.items():
            if r_job.start < cur_timestamp + job.runtime:
                r_job_cpu_per_node = r_job.num_cpus / r_job.nodes
                for n_idx, gpu_list in r_job.allocated_gpus.items():
                    for gpu_idx in gpu_list:
                        if not self.nodes[n_idx].gpu_dict[gpu_idx]:
                            node_free_gpu_list[n_idx].remove(gpu_idx)
                node_free_cpu_count[n_idx] -= r_job_cpu_per_node

        node_free_gpu_count = [len(g) for g in node_free_gpu_list]

        node_free_count = [(i, node_free_gpu_count[i], node_free_cpu_count[i])
                           for i in range(len(node_free_gpu_count))]
        if self.binpack == 'first-fit':
            pass
        elif self.binpack == 'best-fit':
            # Sort by nodes with the least free GPU(s).
            node_free_count.sort(key=lambda x: x[1])
        elif self.binpack == 'worst-fit':
            # Sort by nodes with the most free GPU(s). Don't use, very bad.
            node_free_count.sort(key=lambda x: x[1], reverse=True)
        elif self.binpack == 'tetris':
            # Sort nodes by the most free in terms of "normalized" dot product of free node resources and job resources (multi resource setting).
            pass
        else:
            raise ValueError(f'Invalid allocation strategy {self.binpack}!')

        # Maps node idx to list of gpu indexes for the job to take.
        temp = False
        node_idx_taken = {}
        for list_idx, gpu_demand in enumerate(list(job_gpu_demands)):
            for n_idx, free_gpus, free_cpus in node_free_count:
                if n_idx in node_idx_taken:
                    continue
                if free_gpus >= gpu_demand:
                    if free_cpus >= num_cpus_per_node:
                        # TODO: Reserved GPUs in the beginning of list. Prioritize taking reserved.
                        node_idx_taken[n_idx] = node_free_gpu_list[
                                                    n_idx][:gpu_demand]
                        job_gpu_demands.remove(gpu_demand)
                        break
                    else:
                        pass
                        # if job.num_gpus > 0 and not temp:
                        #     blocked_by_gpu_cpu_job += 1
                        #     temp = True

        # If there are still demands that cannot be satisifed via free and preempted jobs,
        # it cannot be scheduled on the cluster.
        if job_gpu_demands:
            # if temp:
            #     print(
            #         f'GPU-CPU block occurrences: {blocked_by_gpu_cpu_job}, CPU block occurrences: {blocked_by_cpu_job}'
            #     )
            return False, []

        # =============================================================================
        # Execute Job Plans
        # =============================================================================
        # Job plan stores in `node_idx_taken`: {Node Index -> List of GPU Indexes}
        for n_idx, gpu_demand_list in node_idx_taken.items():
            node = self.nodes[n_idx]
            # Verify resource availability before modification
            if node.free_gpus < len(gpu_demand_list) or node.free_cpus < num_cpus_per_node:
                # Rollback or fail gracefully instead of crashing
                # For simulation stability, we can return False here if unexpected contention occurs
                # But since we're single-threaded in simulation logic, this shouldn't happen unless logic error
                # Let's add a safe check
                print(f"Warning: Resource contention on Node {n_idx}. Free GPUs: {node.free_gpus}, Demand: {len(gpu_demand_list)}")
                return False, []
                
            node.free_gpus -= len(gpu_demand_list)
            node.free_cpus -= num_cpus_per_node
            
            if node.free_gpus < 0 or node.free_cpus < 0:
                # Should be caught by check above, but as a fallback
                raise ValueError(f'Ran out of cluster resources on Node {n_idx}!')
                
            for idx in gpu_demand_list:
                if node.gpu_dict[idx] is not None:
                    raise ValueError('Generated execution plan is incorrect.')
                node.gpu_dict[idx] = job
            job.allocated_gpus[n_idx] = gpu_demand_list
        job.start = cur_timestamp
        self.active_jobs[job.idx] = job

        return True, []

    def _get_rl_state(self, job=None):
        """获取环境状态向量，用于RL决策"""
        # 构建状态向量
        # 1. 集群状态 - 每个节点的GPU和CPU使用情况
        cluster_state = []
        for node in self.nodes:
            # 节点GPU利用率
            gpu_util = (self.num_gpus_per_node - node.free_gpus) / self.num_gpus_per_node
            # 节点CPU利用率
            cpu_util = (self.num_cpus_per_node - node.free_cpus) / self.num_cpus_per_node
            cluster_state.extend([gpu_util, cpu_util])

        # 如果没有正在考虑的作业，用零填充
        if job is None:
            job_state = [0, 0]
        else:
            # 2. 作业特征 - 归一化的资源需求
            gpu_demand = job.resources['GPUs'] / (self.num_gpus_per_node * 2)  # 最多需要2个节点的GPU
            cpu_demand = job.resources['CPUs'] / (self.num_cpus_per_node * 2)  # 最多需要2个节点的CPU
            job_state = [gpu_demand, cpu_demand]

        # 组合状态
        state = cluster_state + job_state

        # 确保状态维度为130
        if len(state) < 130:
            state = state + [0] * (130 - len(state))
        elif len(state) > 130:
            state = state[:130]

        return np.array(state, dtype=np.float32)

    def _calculate_enhanced_reward(self, success, job, current_time, action, wait_factor=1.0, system_load=1.0):
        """改进的奖励函数，考虑节点间负载均衡"""
        # 基础奖励
        reward = 0.0

        if success:
            # 成功分配到集群，给予正奖励
            self.successful_allocations += 1

            # 计算利用率提升
            total_gpus = self.num_nodes * self.num_gpus_per_node
            used_gpus = sum([node.num_gpus_per_node - node.free_gpus for node in self.nodes])
            utilization = used_gpus / total_gpus

            # 根据利用率和等待时间调整奖励
            reward += 1.0 + 0.5 * utilization

            # 如果是大作业且成功分配，额外奖励
            if job.resources['GPUs'] > self.num_gpus_per_node / 2:
                reward += 0.3

            # 如果等待时间合理，额外奖励
            waiting_time = current_time - job.arrival
            if waiting_time <= job.runtime * 0.5:
                reward += 0.2

            # 新增：计算节点负载均衡因子
            node_utilizations = []
            for node in self.nodes:
                # 使用节点自己的属性
                gpu_util = (node.num_gpus_per_node - node.free_gpus) / node.num_gpus_per_node
                cpu_util = (node.num_cpus_per_node - node.free_cpus) / node.num_cpus_per_node
                # 综合利用率
                node_util = (gpu_util + cpu_util) / 2
                node_utilizations.append(node_util)

            # 计算节点间利用率方差，值越小表示越均衡
            if len(node_utilizations) > 1:
                load_variance = np.var(node_utilizations)

                # 负载均衡奖励 - 方差越小，奖励越大
                balance_coefficient = 0.3 * min(3.0, system_load)  # 高负载时更加重视均衡
                balance_reward = balance_coefficient * (1.0 - load_variance)
                reward += balance_reward

                # 额外奖励高负载下的有效调度
                if system_load >= 2.0 and utilization > 0.85:
                    reward += 0.2 * system_load  # 高负载下维持高利用率给予额外奖励
        else:
            # 分配失败，基础负奖励
            reward = -0.5

            # 如果是小作业分配失败，额外惩罚
            if job.resources['GPUs'] <= self.num_gpus_per_node / 4:
                reward -= 0.3

            # 如果发送到云上且是大作业，成本高，额外惩罚
            if action == self.num_nodes and job.resources['GPUs'] > self.num_gpus_per_node / 2:
                reward -= 0.2 * job.resources['GPUs'] / self.num_gpus_per_node

        # 计算稳定性奖励（保留原有部分）
        if hasattr(self, 'action_history') and len(self.action_history) > 0:
            recent_actions = self.action_history[-5:]
            if action in recent_actions:
                stability_reward = 0.1 * recent_actions.count(action) / len(recent_actions)
                reward += stability_reward

            if len(recent_actions) >= 3:
                changes = sum(1 for i in range(1, len(recent_actions)) if recent_actions[i] != recent_actions[i - 1])
                if changes >= 3:
                    reward -= 0.15

        # 更新动作历史
        if not hasattr(self, 'action_history'):
            self.action_history = []
        self.action_history.append(action)
        if len(self.action_history) > 20:
            self.action_history.pop(0)

        # 记录决策次数
        self.decisions_made += 1

        return reward

    def predict_wait(self, cur_timestamp, job, queue, loop=False):
        max_timestamp = job.deadline - job.runtime

        num_gpus = job.num_gpus

        def get_gpu_demand_list(cur_job):
            num_gpus = job.num_gpus
            if job.nodes == 1:
                if num_gpus > self.num_gpus_per_node:
                    # Assume worst case colocation
                    # Multinode case, i.e. 26 GPUs, 8 GPU/node cluster -> job_gpu_demands = [8,8,8,2]
                    job_gpu_demands = [self.num_gpus_per_node] * int(
                        num_gpus / self.num_gpus_per_node)
                    if num_gpus % self.num_gpus_per_node:
                        job_gpu_demands.append(num_gpus %
                                               self.num_gpus_per_node)
                else:
                    job_gpu_demands = [num_gpus]
            else:
                job_gpu_demands = [int(num_gpus / job.nodes)] * job.nodes
            return job_gpu_demands

        job_gpu_demands = get_gpu_demand_list(job)

        # Generate job GPU demands
        if job.nodes == 1:
            if num_gpus > self.num_gpus_per_node:
                # Assume worst case colocation
                # Multinode case, i.e. 26 GPUs, 8 GPU/node cluster -> job_gpu_demands = [8,8,8,2]
                job_gpu_demands = [self.num_gpus_per_node] * int(
                    num_gpus / self.num_gpus_per_node)
                if num_gpus % self.num_gpus_per_node:
                    job_gpu_demands.append(num_gpus % self.num_gpus_per_node)
            else:
                job_gpu_demands = [num_gpus]
        else:
            job_gpu_demands = [int(num_gpus / job.nodes)] * job.nodes

        node_free_gpu_count = [0] * self.num_nodes

        for n_idx, node in enumerate(self.nodes):
            for gpu_idx in range(self.num_gpus_per_node):
                if node.gpu_dict[gpu_idx]:
                    continue
                node_free_gpu_count[n_idx] += 1

        # "Plan" ahead of the queue
        # for q_job in queue:
        #     q_job_gpu_demands = get_gpu_demand_list(q_job)
        #     q_node_index = can_cluster_fit(node_free_gpu_count)

        active_job_list = [a_job for a_job in self.active_jobs.values()]
        active_job_list.sort(key=lambda x: x.start + x.runtime)

        def can_cluster_fit(free_gpu_count):
            node_indexes = []
            for demand_idx, job_gpu_demand in enumerate(job_gpu_demands):
                for node_idx, free_gpus_node in enumerate(free_gpu_count):
                    if job_gpu_demand <= free_gpus_node \
                            and node_idx not in node_indexes:
                        node_indexes.append(node_idx)
                    if len(node_indexes) == len(job_gpu_demands):
                        return node_indexes
            if len(node_indexes) != len(job_gpu_demands):
                return []
            return node_indexes

        if can_cluster_fit(node_free_gpu_count):
            return True

        for a_job in active_job_list:
            if a_job.start + a_job.runtime > max_timestamp:
                return False
            for n_idx, gpu_list in a_job.allocated_gpus.items():
                node_free_gpu_count[n_idx] += len(gpu_list)

            if can_cluster_fit(node_free_gpu_count):
                return True
        return False

    # Backfill Scheduling: Reserve blocking job.
    def try_reserve(self, cur_timestamp, job):
        max_timestemp = job.deadline - job.runtime

        free_gpus = [n.free_gpus for n in self.nodes]
        active_job_list = [a_job for a_job in self.active_jobs.values()]
        active_job_list.sort(key=lambda x: x.start + x.runtime)

        num_gpus = job.num_gpus
        # Generate job GPU demands
        if num_gpus > self.num_gpus_per_node:
            # Multinode case, i.e. 26 GPUs, 8 GPU/node cluster -> job_gpu_demands = [8,8,8,2]
            job_gpu_demands = [self.num_gpus_per_node] * int(
                num_gpus / self.num_gpus_per_node)
            if num_gpus % self.num_gpus_per_node:
                job_gpu_demands.append(num_gpus % self.num_gpus_per_node)
        else:
            job_gpu_demands = [num_gpus]

        node_free_list = [[] for _ in range(self.num_nodes)]
        node_free_count = [0] * self.num_nodes
        for n_idx, node in enumerate(self.nodes):
            for gpu_idx in range(self.num_gpus_per_node):
                if node.gpu_dict[gpu_idx] or node.reserved_gpus[gpu_idx]:
                    continue
                node_free_count[n_idx] += 1
                node_free_list[n_idx].append(gpu_idx)

        for a_job in active_job_list:
            if a_job.start + a_job.runtime > job.deadline - job.runtime:
                return False
            for n_idx, gpu_list in a_job.allocated_gpus.items():
                for gpu_idx in gpu_list:
                    if self.nodes[n_idx].reserved_gpus[gpu_idx]:
                        continue
                    node_free_list[n_idx].append(gpu_idx)
                    node_free_count[n_idx] += 1

            node_indexes = utils.is_subset(node_free_count, job_gpu_demands)
            if node_indexes:
                for idx, n_idx in enumerate(node_indexes):
                    gpu_list = node_free_list[n_idx][-job_gpu_demands[idx]:]
                    job.allocated_gpus[n_idx] = gpu_list
                    cur_node = self.nodes[n_idx]
                    for gpu_idx in gpu_list:
                        cur_node.reserved_gpus[gpu_idx] = job
                self.reserved_jobs[job.idx] = job
                job.block_job_idx = a_job.idx
                job.start = a_job.start + a_job.runtime
                return True
        raise ValueError('I should not go here!')

    def try_clear(self, t: float):
        """Clears cluster of completed jobs at time t.
        """
        completed_jobs = []
        # Free jobs on the cluster which have completed.
        for job_idx, job in list(self.active_jobs.items()):
            # If job has finished before time t...
            if t >= job.start + job.runtime:
                for node_idx, gpu_list in job.allocated_gpus.items():
                    cur_node = self.nodes[node_idx]
                    node_gpu_dict = cur_node.gpu_dict
                    for gpu_idx in gpu_list:
                        node_gpu_dict[gpu_idx] = None
                    cur_node.free_gpus += len(gpu_list)
                    cur_node.free_cpus += job.num_cpus / job.nodes
                completed_jobs.append(job)

        # Clears cluster of completed jobs.
        for job in completed_jobs:
            job.state = 'LOCAL'
            del self.active_jobs[job.idx]

        return completed_jobs

    def __repr__(self):
        repr_str = 'Cluster State:\n'
        for idx, n in enumerate(self.nodes):
            repr_str += f'Node {idx}: {n}\n'
        return repr_str

    def _execute_allocation(self, node_idx, job, cur_timestamp):
        """在指定节点上尝试分配作业，支持RL调度决策

        Args:
            node_idx: 节点索引
            job: 要分配的作业
            cur_timestamp: 当前时间戳

        Returns:
            bool: 分配是否成功
        """
        # 检查节点索引是否有效
        if node_idx < 0 or node_idx >= self.num_nodes:
            return False

        node = self.nodes[node_idx]

        # 检查资源是否足够
        if node.free_gpus < job.resources['GPUs'] or node.free_cpus < job.resources['CPUs']:
            return False

        # 找到可用GPU
        gpu_list = []
        for gpu_idx, gpu_job in node.gpu_dict.items():
            if gpu_job is None and len(gpu_list) < job.resources['GPUs']:
                gpu_list.append(gpu_idx)

        if len(gpu_list) < job.resources['GPUs']:
            return False

        # 分配资源
        for gpu_idx in gpu_list:
            node.gpu_dict[gpu_idx] = job

        node.free_gpus -= job.resources['GPUs']
        node.free_cpus -= job.resources['CPUs']

        # 更新作业状态
        job.allocated_gpus = {node_idx: gpu_list}
        job.start = cur_timestamp
        self.active_jobs[job.idx] = job

        return True

