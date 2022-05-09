from copy import deepcopy
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import wandb
import pprint

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from train.train_script import *
from datasets.caueeg_script import *
from models.utils import count_parameters


def check_device_env(cfg_default):
    if not torch.cuda.is_available():
        raise ValueError('ERROR: No GPU is available. Check the environment again!!')

    # assign GPU
    cfg_default['device'] = torch.device(cfg_default.get('device', 'cuda')
                                         if torch.cuda.is_available() else 'cpu')

    # set the minibatch size according to the GPU memory
    device_name = torch.cuda.get_device_name(0)
    if '3090' in device_name:
        cfg_default['minibatch'] = 160
    elif '2080' in device_name:
        cfg_default['minibatch'] = 96
    elif '1070' in device_name:
        cfg_default['minibatch'] = 64

    # distributed training
    if cfg_default.get('ddp', False):
        world_size = torch.cuda.device_count()
        if world_size > 1:
            cfg_default['ddp_size'] = cfg_default.get('ddp_size', world_size)
        else:
            raise ValueError(f'ERROR: There are not sufficient GPUs to launch the DDP training: {world_size}. '
                             f'Check the environment again!!')


def prepare_and_run_train(rank, world_size, config):
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

    # generate the model and update some configurations
    model = hydra.utils.instantiate(config)
    if use_ddp:
        model = DDP(model, device_ids=[config['device']])
    else:
        model = model.to(config['device'])

    config['output_length'] = model.get_output_length()
    config['num_params'] = count_parameters(model)

    # train
    model = train_with_wandb(config, model, train_loader, val_loader, test_loader, multicrop_test_loader,
                             config['preprocess_train'], config['preprocess_test'])

    # cleanup
    if use_ddp:
        torch.distributed.destroy_process_group()


@hydra.main(config_path='config', config_name='default')
def my_app(cfg: DictConfig) -> None:
    # initialize the configurations
    # print(OmegaConf.to_yaml(cfg))
    cfg_default = {**OmegaConf.to_container(cfg.data), **OmegaConf.to_container(cfg.train),
                   **OmegaConf.to_container(cfg.model), 'cwd': HydraConfig.get().runtime.cwd}

    # check the workstation environment and update some configurations
    check_device_env(cfg_default)

    # initialize the wandb
    wandb_run = wandb.init(project=f"{cfg_default['project']}")
    wandb.run.name = wandb.run.id

    with wandb_run:
        config = {}

        # load default configurations not selected by wandb.sweep
        for k, v in cfg_default.items():
            if k not in [wandb_key.split('.')[-1] for wandb_key in wandb.config.keys()]:
                config[k] = v

        # load the selected configurations from wandb sweep with preventing callables from type-conversion to str
        for k, v in wandb.config.items():
            k = k.split('.')[-1]
            if k not in config:
                config[k] = v

        # build the dataset and train the model
        if config.get('ddp', False):
            mp.spawn(prepare_and_run_train,
                     args=(config['ddp_size'], config,),
                     nprocs=config['ddp_size'],
                     join=True)
        else:
            prepare_and_run_train(rank=None, world_size=None, config=config)


if __name__ == "__main__":
    my_app()
