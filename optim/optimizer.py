import torch.optim as optim


def get_optimizer(model, config):
    optimizer = config.get("optimizer", "adamw").lower()
    if optimizer == "sgd":
        config["momentum"] = config.get("momentum", 0.9)
        return optim.SGD(
            model.parameters(),
            lr=config["base_lr"],
            weight_decay=config["weight_decay"],
            momentum=config["momentum"],
        )
    elif optimizer == "adam":
        return optim.Adam(
            model.parameters(),
            lr=config["base_lr"],
            weight_decay=config["weight_decay"],
        )
    elif optimizer == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config["base_lr"],
            weight_decay=config["weight_decay"],
        )
