import os
from copy import deepcopy
import gc
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from train.train_script import train_script
from datasets.caueeg_script import build_dataset_for_train
from models.utils import count_parameters


def check_device_env(config):
    if not torch.cuda.is_available():
        raise ValueError('ERROR: No GPU is available. Check the environment again!!')

    # assign GPU
    config['device'] = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')

    # set the minibatch size according to the GPU memory
    device_name = torch.cuda.get_device_name(0)
    if '3090' in device_name:
        pass
    elif '2080' in device_name:
        config['minibatch'] = config['minibatch'] // 2
    elif '1070' in device_name:
        config['minibatch'] = config['minibatch'] // 4

    # distributed training
    if config.get('ddp', False):
        world_size = torch.cuda.device_count()
        if world_size > 1:
            config['ddp_size'] = config.get('ddp_size', world_size)
        else:
            raise ValueError(f'ERROR: There are not sufficient GPUs to launch the DDP training: {world_size}. '
                             f'Check the environment again!!')


def prepare_and_run_train(rank, world_size, config):
    # collect some garbage
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # fix the seed for reproducibility (a negative seed value means not fixing)
    if config.get('seed', 0) >= 0:
        seed = config.get('seed', 0)
        seed = seed + rank if rank is not None else seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    # setup for distributed training
    use_ddp = config.get('ddp', False)

    if use_ddp:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
        config = deepcopy(config)
        config['device'] = rank

    # compose dataset
    train_loader, val_loader, test_loader, multicrop_test_loader = build_dataset_for_train(config)

    from torchsummaryX import summary
    summary
    # generate the model and update some configurations
    model = hydra.utils.instantiate(config)

    if use_ddp:
        torch.cuda.set_device(config['device'])
        model.cuda(config['device'])
        model = DDP(model, device_ids=[config['device']])
        config['output_length'] = model.module.get_output_length()
        config['num_params'] = count_parameters(model)
        torch.distributed.barrier()
    else:
        model = model.to(config['device'])
        config['output_length'] = model.get_output_length()
        config['num_params'] = count_parameters(model)

    # train
    train_script(config, model, train_loader, val_loader, test_loader, multicrop_test_loader,
                 config['preprocess_train'], config['preprocess_test'])

    # cleanup
    if use_ddp:
        torch.distributed.destroy_process_group()


@hydra.main(config_path='config', config_name='default')
def my_app(cfg: DictConfig) -> None:
    # initialize the configurations
    # print(OmegaConf.to_yaml(cfg))
    config = {**OmegaConf.to_container(cfg.data), **OmegaConf.to_container(cfg.train),
              **OmegaConf.to_container(cfg.model), 'cwd': HydraConfig.get().runtime.cwd}

    # check the workstation environment and update some configurations
    check_device_env(config)

    # build the dataset and train the model
    if config.get('ddp', False):
        mp.spawn(prepare_and_run_train,
                 args=(config['ddp_size'], config),
                 nprocs=config['ddp_size'],
                 join=True)
    else:
        prepare_and_run_train(rank=None, world_size=None, config=config)


if __name__ == "__main__":
    my_app()
