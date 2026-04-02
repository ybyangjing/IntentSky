class Node(object):
    def __init__(self, num_gpus_per_node, num_cpus_per_node):
        # 添加节点总GPU和CPU数量作为实例属性
        self.num_gpus_per_node = num_gpus_per_node
        self.num_cpus_per_node = num_cpus_per_node
        
        # 当前空闲资源
        self.free_gpus = num_gpus_per_node
        self.free_cpus = num_cpus_per_node
        
        # 资源分配状态跟踪
        self.gpu_dict = {i: None for i in range(num_gpus_per_node)}
        self.reserved_gpus = {i: None for i in range(num_gpus_per_node)}
    
    def __repr__(self):
        return f'GPUs: {self.free_gpus}/{self.num_gpus_per_node}, CPUs: {self.free_cpus}/{self.num_cpus_per_node}'
