import os

import torch.multiprocessing
import wandb

from text.train_functions_ddp import train_function

torch.multiprocessing.set_sharing_strategy('file_system')


gpus = 4
nodes = 1
node_rank = 0
world_size = 4
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '8888'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':64:8'
os.environ['NCCL_LL_THRESHOLD'] = '0.'
os.environ['NCCL_DEBUG'] = 'INFO'
# os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
# os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['WANDB_SILENT'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

if __name__ == '__main__':
    group_name = wandb.util.generate_id()

    for i in range(5):
        torch.multiprocessing.spawn(
            fn=train_function, nprocs=gpus,
            args=(world_size, node_rank, gpus, i, group_name),
            join=True)
