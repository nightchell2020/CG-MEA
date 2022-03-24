import torch
from torch.utils.tensorboard import SummaryWriter

# __all__ = []


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_final_shape(model, train_loader, device):
    for sample_batched in train_loader:
        x = torch.zeros_like(sample_batched['signal']).to(device)
        model(x, age=sample_batched['age'].to(device))
        return model.get_final_shape()


def visualize_network_tensorboard(model, train_loader, device, nb_fname, name):
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/' + nb_fname + '_' + name)

    for batch_i, sample_batched in enumerate(train_loader):
        # pull up the batch data
        x = sample_batched['signal'].to(device)
        age = sample_batched['age'].to(device)

        # apply model on whole batch directly on device
        writer.add_graph(model, (x, age))
        break

    writer.close()

