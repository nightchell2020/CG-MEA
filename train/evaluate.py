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


@torch.no_grad()
def check_accuracy(model, loader, preprocess, config, repeat=1):
    model.eval()

    # for accuracy
    correct, total = (0, 0)

    with torch.no_grad():
        for k in range(repeat):
            for sample_batched in loader:
                # preprocessing (this includes to-device operation)
                preprocess(sample_batched)

                # apply model on whole batch directly on device
                x = sample_batched['signal']
                age = sample_batched['age']
                y = sample_batched['class_label']
                output = model(x, age)

                if config['criterion'] == 'cross-entropy':
                    s = F.softmax(output, dim=1)
                elif config['criterion'] == 'multi-bce':
                    s = torch.sigmoid(output)

                # calculate accuracy
                pred = s.argmax(dim=-1)
                correct += pred.squeeze().eq(y).sum().item()
                total += pred.shape[0]

    accuracy = 100.0 * correct / total
    return accuracy


@torch.no_grad()
def check_accuracy_extended(model, loader, preprocess, config, repeat=1):
    model.eval()

    # for accuracy
    correct, total = (0, 0)

    # for confusion matrix
    C = config['out_dims']
    confusion_matrix = np.zeros((C, C), dtype=np.int32)

    # for error table
    error_table = {data['serial']: {'GT': data['class_label'].item(),
                                    'Pred': [0] * C} for data in loader.dataset}

    # for ROC curve
    score = None
    target = None

    with torch.no_grad():
        for k in range(repeat):
            for sample_batched in loader:
                # preprocessing (this includes to-device operation)
                preprocess(sample_batched)

                # apply model on whole batch directly on device
                x = sample_batched['signal']
                age = sample_batched['age']
                y = sample_batched['class_label']
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

                # error table
                for n in range(pred.shape[0]):
                    serial = sample_batched['serial'][n]
                    error_table[serial]['Pred'][pred[n].item()] += 1

    # error table update
    error_table_serial = []
    error_table_pred = []
    error_table_gt = []

    for serial in sorted(error_table.keys()):
        error_table_serial.append(serial)
        error_table_pred.append(error_table[serial]['Pred'])
        error_table_gt.append(error_table[serial]['GT'])

    error_table = {'Serial': error_table_serial,
                   'Pred': error_table_pred,
                   'GT': error_table_gt}

    accuracy = 100.0 * correct / total
    return accuracy, score, target, confusion_matrix, error_table


@torch.no_grad()
def check_accuracy_extended_debug(model, loader, preprocess, config, repeat=1):
    model.eval()

    # for accuracy
    correct, total = (0, 0)

    # for confusion matrix
    C = config['out_dims']
    confusion_matrix = np.zeros((C, C), dtype=np.int32)

    # for error table
    error_table = {data['serial']: {'GT': data['class_label'].item(),
                                    'Pred': [0] * C} for data in loader.dataset}

    # for crop timing
    crop_timing = dict()

    # for ROC curve
    score = None
    target = None

    with torch.no_grad():
        for k in range(repeat):
            for sample_batched in loader:
                # preprocessing (this includes to-device operation)
                preprocess(sample_batched)

                # apply model on whole batch directly on device
                x = sample_batched['signal']
                age = sample_batched['age']
                y = sample_batched['class_label']
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

                # error table
                for n in range(pred.shape[0]):
                    serial = sample_batched['serial'][n]
                    error_table[serial]['Pred'][pred[n].item()] += 1

                # crop timing
                for n in range(pred.shape[0]):
                    ct = sample_batched['crop_timing'][n]

                    if ct not in crop_timing.keys():
                        crop_timing[ct] = {}

                    if pred[n] == y[n]:
                        crop_timing[ct]['correct'] = crop_timing[ct].get('correct', 0) + 1
                    else:
                        crop_timing[ct]['incorrect'] = crop_timing[ct].get('incorrect', 0) + 1

    # error table update
    error_table_serial = []
    error_table_pred = []
    error_table_gt = []

    for serial in sorted(error_table.keys()):
        error_table_serial.append(serial)
        error_table_pred.append(error_table[serial]['Pred'])
        error_table_gt.append(error_table[serial]['GT'])

    error_table = {'Serial': error_table_serial,
                   'Pred': error_table_pred,
                   'GT': error_table_gt}

    accuracy = 100.0 * correct / total
    return accuracy, score, target, confusion_matrix, error_table, crop_timing


@torch.no_grad()
def check_accuracy_multicrop(model, loader, preprocess, config, repeat=1):
    model.eval()

    # for accuracy
    correct, total = (0, 0)

    with torch.no_grad():
        for k in range(repeat):
            for sample_batched in loader:
                # preprocessing (this includes to-device operation)
                preprocess(sample_batched)

                # apply model on whole batch directly on device
                x = sample_batched['signal']
                age = sample_batched['age']
                y = sample_batched['class_label']
                output = model(x, age)

                if config['criterion'] == 'cross-entropy':
                    s = F.softmax(output, dim=1)
                elif config['criterion'] == 'multi-bce':
                    s = torch.sigmoid(output)

                # multi-crop averaging
                if s.size(0) % config['test_crop_multiple'] != 0:
                    raise ValueError(f"check_accuracy_multicrop(): Real minibatch size={x.size(0)} is not multiple of"
                                     f"config['test_crop_multiple']={config['test_crop_multiple']}.")

                real_minibatch = s.size(0) // config['test_crop_multiple']
                s_ = torch.zeros((real_minibatch, s.size(1)))
                y_ = torch.zeros((real_minibatch,), dtype=torch.int32)

                for m in range(real_minibatch):
                    s_[m] = s[config['test_crop_multiple']*m:config['test_crop_multiple']*(m + 1)].mean(dim=0, keepdims=True)
                    y_[m] = y[config['test_crop_multiple']*m]

                s = s_
                y = y_

                # calculate accuracy
                pred = s.argmax(dim=-1)
                correct += pred.squeeze().eq(y).sum().item()
                total += pred.shape[0]

    accuracy = 100.0 * correct / total

    return accuracy


@torch.no_grad()
def check_accuracy_multicrop_extended(model, loader, preprocess, config, repeat=1):
    model.eval()

    # for accuracy
    correct, total = (0, 0)

    # for confusion matrix
    C = config['out_dims']
    confusion_matrix = np.zeros((C, C), dtype=np.int32)

    # for error table
    error_table = {data['serial']: {'GT': data['class_label'].item(),
                                    'Pred': [0] * C} for data in loader.dataset}

    # for crop timing
    crop_timing = dict()

    # for ROC curve
    score = None
    target = None

    with torch.no_grad():
        for k in range(repeat):
            for sample_batched in loader:
                # preprocessing (this includes to-device operation)
                preprocess(sample_batched)

                # apply model on whole batch directly on device
                x = sample_batched['signal']
                age = sample_batched['age']
                y = sample_batched['class_label']
                output = model(x, age)

                if config['criterion'] == 'cross-entropy':
                    s = F.softmax(output, dim=1)
                elif config['criterion'] == 'multi-bce':
                    s = torch.sigmoid(output)

                # multi-crop averaging
                if s.size(0) % config['test_crop_multiple'] != 0:
                    raise ValueError(f"check_accuracy_multicrop(): Real minibatch size={x.size(0)} is not multiple of "
                                     f"config['test_crop_multiple']={config['test_crop_multiple']}.")

                real_minibatch = s.size(0) // config['test_crop_multiple']
                s_ = torch.zeros((real_minibatch, s.size(1)))
                y_ = torch.zeros((real_minibatch,), dtype=torch.int32)
                serial_ = []
                
                for m in range(real_minibatch):
                    s_[m] = s[config['test_crop_multiple']*m:config['test_crop_multiple']*(m + 1)].mean(dim=0, keepdims=True)
                    y_[m] = y[config['test_crop_multiple']*m]
                    serial_.append(sample_batched['serial'][config['test_crop_multiple']*m])

                s = s_
                y = y_

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

                # error table
                for n in range(pred.shape[0]):
                    serial = serial_[n]
                    error_table[serial]['Pred'][pred[n].item()] += 1

    # error table update
    error_table_serial = []
    error_table_pred = []
    error_table_gt = []

    for serial in sorted(error_table.keys()):
        error_table_serial.append(serial)
        error_table_pred.append(error_table[serial]['Pred'])
        error_table_gt.append(error_table[serial]['GT'])

    error_table = {'Serial': error_table_serial,
                   'Pred': error_table_pred,
                   'GT': error_table_gt}

    accuracy = 100.0 * correct / total
    return accuracy, score, target, confusion_matrix, error_table
