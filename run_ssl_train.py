import gc
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from train.ssl_train_script import ssl_train_script

from run_train import check_device_env
from run_train import set_seed
from run_train import initialize_ddp
from run_train import compose_dataset
from run_train import load_pretrained_params
from models.utils import count_parameters


def generate_ssl_model(config):
    # Instantiate backbone
    backbone = hydra.utils.instantiate(config)

    # Count parameters and set output length
    config["num_params"] = count_parameters(backbone)
    config["output_length"] = backbone.get_output_length()

    # ssl embedding layer
    if config["embedding_layer"] == "pool":
        config["embedding_layer"] = config["fc_stages"]
    else:
        config["embedding_layer"] = abs(config["embedding_layer"])

    # Instantiate SSL model
    temp_target = config["_target_"]
    config["_target_"] = config["_ssl_target_"]
    model = hydra.utils.instantiate(config, backbone)
    config["_target_"] = temp_target

    # Set device and enable Distributed Data Parallel if necessary
    device = config["device"]
    if config.get("ddp", False):
        torch.cuda.set_device(device)
        model.cuda(config["device"])
        model = DDP(model, device_ids=[device], find_unused_parameters=True)
        torch.distributed.barrier()
    else:
        model = model.to(device)

    return model


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
    model = generate_ssl_model(config)

    # load pretrained model if needed
    if "load_pretrained" in config.keys():
        load_pretrained_params(model, config)

    # train
    ssl_train_script(config, model, train_loader, config["preprocess_train"])

    # cleanup
    if config.get("ddp", False):
        torch.distributed.destroy_process_group()


@hydra.main(version_base="1.1", config_path="config", config_name="default")
def my_app(cfg: DictConfig) -> None:
    # initialize the configurations
    # print(OmegaConf.to_yaml(cfg))
    config = {
        **OmegaConf.to_container(cfg.data),
        **OmegaConf.to_container(cfg.train),
        **OmegaConf.to_container(cfg.ssl),
        **OmegaConf.to_container(cfg.model),
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
