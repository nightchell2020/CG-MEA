import os
from copy import deepcopy
import gc
from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict
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

    device_name = torch.cuda.get_device_name(0)
    # minibatch size
    if 'minibatch' not in config:
        # set the minibatch size according to the GPU memory
        if '3090' in device_name:
            config['minibatch'] = config['minibatch_3090']
        elif '2080' in device_name:
            config['minibatch'] = config['minibatch_3090'] // 2
        elif '1080' in device_name:
            config['minibatch'] = config['minibatch_3090'] // 4
        elif '1070' in device_name:
            config['minibatch'] = config['minibatch_3090'] // 4

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
        config['device'] = torch.device(f'cuda:{rank}')

    # compose dataset
    train_loader, val_loader, test_loader, multicrop_test_loader = build_dataset_for_train(config)

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

    # load pretrained model if needed
    if 'pretrain' in config.keys():
        save_path = os.path.join(config.get('cwd', ''), f'local/checkpoint/{config["pretrain"]}/')
        ckpt = torch.load(os.path.join(save_path, 'checkpoint.pt'), map_location=config['device'])

        if ckpt['config']['ddp'] == config['ddp']:
            model.load_state_dict(ckpt['model_state'])
        elif ckpt['config']['ddp']:
            model_state_ddp = deepcopy(ckpt['model_state'])
            model_state = OrderedDict()
            for k, v in model_state_ddp.items():
                name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
                model_state[name] = v
            model.load_state_dict(model_state)
        else:
            model.module.load_state_dict(ckpt['model_state'])

    # load teacher network if needed
    if 'distil_teacher' in config.keys():
        # load teacher model
        save_path = os.path.join(config.get('cwd', ''), f'local/checkpoint/{config["distil_teacher"]}/')
        ckpt = torch.load(os.path.join(save_path, 'checkpoint.pt'), map_location=config['device'])
        model_teacher = hydra.utils.instantiate(ckpt['config'])

        if use_ddp:
            model_teacher.cuda(config['device'])
            model_teacher = DDP(model_teacher, device_ids=[config['device']])
            torch.distributed.barrier()
        else:
            model_teacher = model_teacher.to(config['device'])

        if ckpt['config']['ddp'] == config['ddp']:
            model_teacher.load_state_dict(ckpt['model_state'])
        elif ckpt['config']['ddp']:
            model_state_ddp = deepcopy(ckpt['model_state'])
            model_state = OrderedDict()
            for k, v in model_state_ddp.items():
                name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
                model_state[name] = v
            model_teacher.load_state_dict(model_state)
        else:
            model_teacher.module.load_state_dict(ckpt['model_state'])

        model_teacher = model_teacher.requires_grad_(False)
        model_teacher = model_teacher.eval()

        # distill configuration
        config['distil_alpha'] = config.get('distil_alpha', 0.5)
        config['distil_type'] = config.get('distil_type', 'hard')
        if config['distil_type'] == 'soft':
            config['distil_tau'] = config.get('distil_tau', 1.0)
        config['distil_teacher_preprocess'] = ckpt['config']['preprocess_test']
        config['distil_teacher_model'] = model_teacher
        config['distil_teacher_criterion'] = ckpt['config']['criterion']

        # sanity check
        if config['distil_type'] not in ['hard', 'soft']:
            raise ValueError(f"ERROR: Choose the correct option for knowledge distillation: 'soft' or 'hard.'")
        elif config['distil_type'] == 'soft' and config['distil_teacher_criterion'] != config['criterion']:
            raise ValueError(f"ERROR: In the case of 'soft' knowledge distillation, "
                             f"the objective functions must be equal between teacher and student models.\n"
                             f"Current state: teacher - {config['distil_teacher_criterion']}"
                             f" / student - {config['criterion']}")
        elif config['distil_type'] == 'soft' and config['distil_teacher_criterion'] == 'svm':
            raise ValueError(f"ERROR: In our implementation, "
                             f"the SVM classifier does not support for 'soft' knowledge distillation.")
        elif config['EKG'] != ckpt['config']['EKG'] or config['photic'] != ckpt['config']['photic']:
            raise ValueError(f"ERROR: The teacher and student networks must have the same EEG channel configuration:\n"
                             f"Current state: teacher - EKG {ckpt['config']['EKG']}, Photic {ckpt['config']['photic']}"
                             f" / student - EKG {config['EKG']}, Photic {config['photic']}.")

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
