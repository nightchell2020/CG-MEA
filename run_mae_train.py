import os
import gc
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict
import hydra
from hydra.core.hydra_config import HydraConfig

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from train.ssl_train_script import ssl_train_script

from run_train import check_device_env
from run_train import generate_model
from run_train import set_seed
from run_train import initialize_ddp
from run_train import compose_dataset


def load_pretrained_mae(model, config):
    save_path = os.path.join(config.get("cwd", ""), f'local/checkpoint/{config["load_pretrained"]}/')
    ckpt = torch.load(os.path.join(save_path, "checkpoint.pt"), map_location=config["device"])

    # load pretrained model
    if not config.get("ddp", False):
        pre_model_state = ckpt["ssl_model_state"]
    else:
        pre_model_state_ddp = deepcopy(ckpt["ssl_model_state"])
        pre_model_state = OrderedDict()
        for k, v in pre_model_state_ddp.items():
            name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
            pre_model_state[name] = v

    model.load_state_dict(pre_model_state)


def prepare_and_run_ssl_train(rank, world_size, config):
    # collect some garbage
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # fix the seed for reproducibility (a negative seed value means not fixing)
    set_seed(config, rank)

    # setup for distributed training
    if config.get("ddp", False):
        config = initialize_ddp(rank, world_size, config)

    # compose dataset
    train_loader, _, _, _ = compose_dataset(config)

    # generate the model and update some configurations
    model = generate_model(config)

    # load pretrained model if needed
    if "load_pretrained" in config.keys():
        load_pretrained_mae(model, config)

    # train
    ssl_train_script(config, model, train_loader, config["preprocess_train"])

    # cleanup
    if config.get("ddp", False):
        torch.distributed.destroy_process_group()


@hydra.main(config_path="config", config_name="default")
def my_app(cfg: DictConfig) -> None:
    # initialize the configurations
    # print(OmegaConf.to_yaml(cfg))
    config = {
        **OmegaConf.to_container(cfg.data),
        **OmegaConf.to_container(cfg.train),
        # **OmegaConf.to_container(cfg.model),
        **OmegaConf.to_container(cfg.ssl),
        "cwd": HydraConfig.get().runtime.cwd,
    }

    # check the workstation environment and update some configurations
    check_device_env(config)

    # build the dataset and train the model
    if config.get("ddp", False):
        mp.spawn(
            prepare_and_run_ssl_train,
            args=(config["ddp_size"], config),
            nprocs=config["ddp_size"],
            join=True,
        )
    else:
        prepare_and_run_ssl_train(rank=None, world_size=None, config=config)


if __name__ == "__main__":
    my_app()
