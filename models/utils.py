from typing import List, Dict

import math
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# __all__ = []


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def program_conv_filters(sequence_length: int, conv_filter_list: List[Dict],
                         output_lower_bound: int, output_upper_bound: int,
                         stride_pool_ratio: float = 3.00, trials: int = 5, class_name: str = ''):
    # desired
    mid = (output_upper_bound - output_lower_bound ) / 2.0
    in_out_ratio = float(sequence_length) / mid

    base_stride = np.power(in_out_ratio / np.prod([cf['kernel_size'] for cf in conv_filter_list], dtype=np.float64),
                           1.0 / len(conv_filter_list))

    for i in range(len(conv_filter_list)):
        cf = conv_filter_list[i]
        if i == 0:
            stride = max(1.0, base_stride * cf['kernel_size']) - 2.0
            cf['pool'] = max(1, round(float(stride) / cf['kernel_size'] * stride_pool_ratio - 1.2))
        else:
            stride = max(1.0, base_stride * cf['kernel_size'])
            cf['pool'] = max(1, round(float(stride) / cf['kernel_size'] * stride_pool_ratio))

        cf['stride'] = round(stride / cf['pool'])
        # cf['dilation'] = 1
        conv_filter_list[i] = cf

    success = False
    str_debug = f"\n{'-'*100}\nstarting from sequence length: {sequence_length}\n{'-'*100}\n"
    current_length = sequence_length

    for k in range(trials):
        if success:
            break

        for pivot in reversed(range(len(conv_filter_list))):
            current_length = sequence_length

            for cf in conv_filter_list:
                current_length = current_length // cf.get('pool', 1)
                str_debug += f"{cf} >> {current_length} "

                effective_kernel_size = (cf['kernel_size'] - 1) * cf.get('dilation', 1)
                both_side_pad = 2 * (cf['kernel_size'] // 2)
                current_length = (current_length + both_side_pad - effective_kernel_size - 1) // cf['stride'] + 1
                str_debug += f">> {current_length}\n"

            if current_length < output_lower_bound:
                conv_filter_list[pivot]['stride'] = max(1, conv_filter_list[pivot]['stride'] - 1)
            elif current_length > output_upper_bound:
                conv_filter_list[pivot]['stride'] += 1
            else:
                str_debug += f">> Success!"
                success = True
                break

            str_debug += f">> Failed.."
            str_debug += f"\n{'-' * 100}\n"

    if not success:
        header = class_name + ', ' if len(class_name) > 0 else ''
        raise ValueError(f"{header}conv1d_filter_programming() failed to determine "
                         f"the proper convolution filter parameters. "
                         f"The following is the recording for debug: {str_debug}")

    # print(str_debug)
    output_length = current_length
    return output_length


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


def make_pool_or_not(base_pool, pool: int):
    def do_nothing(x):
        return x

    if pool == 1:
        return do_nothing
    elif pool > 1:
        return base_pool(pool)
    else:
        raise ValueError(f'make_pool_or_not(pool) receives an invalid value as input.')
