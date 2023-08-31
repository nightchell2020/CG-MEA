import torch.optim as optim


def get_optimizer(model, config):
    optimizer = config.get("optimizer", "adamw").lower()

    if config.get("layer_wise_lr", False):
        parameters = model.layer_wise_lr_params(weight_decay=config["weight_decay"])
    else:
        parameters = model.parameters()

    if optimizer == "sgd":
        config["momentum"] = config.get("momentum", 0.9)
        return optim.SGD(
            parameters,
            lr=config["base_lr"],
            weight_decay=config["weight_decay"],
            momentum=config["momentum"],
        )
    elif optimizer == "adam":
        return optim.Adam(
            parameters,
            lr=config["base_lr"],
            weight_decay=config["weight_decay"],
        )
    elif optimizer == "adamw":
        return optim.AdamW(
            parameters,
            lr=config["base_lr"],
            weight_decay=config["weight_decay"],
        )
