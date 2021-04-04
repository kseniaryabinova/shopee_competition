import os

import torch


import torch.multiprocessing

from image.train_functions_ddp import train_function

torch.multiprocessing.set_sharing_strategy('file_system')


gpus = 4
nodes = 1
node_rank = 0
world_size = 4
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '8888'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':64:8'
os.environ['NCCL_LL_THRESHOLD'] = '0.'
os.environ['WANDB_SILENT'] = 'true'

if __name__ == '__main__':
    torch.multiprocessing.spawn(fn=train_function, nprocs=gpus, args=(world_size, node_rank, gpus), join=True)
