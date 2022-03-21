import numpy as np
import torch
import torch.nn.functional as F

# __all__ = []


def calculate_confusion_matrix(pred, target, num_classes):
    N = target.shape[0]
    C = num_classes
    confusion = np.zeros((C, C), dtype=np.int32)

    for i in range(N):
        r = target[i]
        c = pred[i]
        confusion[r, c] += 1
    return confusion


def check_accuracy(model, loader, config, repeat=1):
    model.eval()
    device = config['device']

    # for accuracy
    correct, total = (0, 0)

    # for confusion matrix
    C = config['out_dims']
    confusion_matrix = np.zeros((C, C), dtype=np.int32)

    # for debug table
    debug_table = {data['metadata']['serial']: {'GT': data['class_label'].item(),
                                                'Acc': 0,
                                                'Pred': [0] * C} for data in loader.dataset}

    # for ROC curve
    score = None
    target = None

    with torch.no_grad():
        for k in range(repeat):
            for sample_batched in loader:
                # pull up the data
                x = sample_batched['signal'].to(device)
                age = sample_batched['age'].to(device)
                y = sample_batched['class_label'].to(device)

                # apply model on whole batch directly on device
                output = model(x, age)

                if config['criterion'] == 'cross-entropy':
                    s = F.softmax(output, dim=1)
                elif config['criterion'] == 'multi-bce':
                    s = torch.sigmoid(output)

                # calculate accuracy
                pred = s.argmax(dim=-1)
                correct += pred.squeeze().eq(y).sum().item()
                total += pred.shape[0]

                if score is None:
                    score = s.detach().cpu().numpy()
                    target = y.detach().cpu().numpy()
                else:
                    score = np.concatenate((score, s.detach().cpu().numpy()), axis=0)
                    target = np.concatenate((target, y.detach().cpu().numpy()), axis=0)

                # confusion matrix
                confusion_matrix += calculate_confusion_matrix(pred, y, num_classes=config['out_dims'])

                # debug table
                for n in range(pred.shape[0]):
                    serial = sample_batched['metadata'][n]['serial']
                    debug_table[serial]['edfname'] = sample_batched['metadata'][n]['edfname']
                    debug_table[serial]['Pred'][pred[n].item()] += 1
                    acc = debug_table[serial]['Pred'][y[n].item()] / np.sum(debug_table[serial]['Pred']) * 100
                    debug_table[serial]['Acc'] = f'{acc:>6.02f}%'

    # debug table update
    debug_table_serial = []
    debug_table_edf = []
    debug_table_pred = []
    debug_table_gt = []

    for key, val in debug_table.items():
        debug_table_serial.append(key)
        debug_table_edf.append(val['edfname'])
        debug_table_pred.append(val['Pred'])
        debug_table_gt.append(val['GT'])

    debug_table = (debug_table_serial, debug_table_edf, debug_table_pred, debug_table_gt)

    accuracy = 100.0 * correct / total
    return accuracy, confusion_matrix, debug_table, score, target
