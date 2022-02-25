from itertools import cycle
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  # roc_auc_score
from sklearn.preprocessing import label_binarize
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
import wandb


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


def train_multistep(model, loader, optimizer, scheduler, config, steps):
    model.train()
    device = config['device']

    i = 0
    cumu_loss = 0
    correct, total = (0, 0)

    while True:
        for sample_batched in loader:
            optimizer.zero_grad()

            # load the mini-batched data
            x = sample_batched['signal'].to(device)
            age = sample_batched['age'].to(device)
            y = sample_batched['class_label'].to(device)

            # forward pass
            output = model(x, age)

            # loss function
            if config['criterion'] == 'cross-entropy':
                s = F.log_softmax(output, dim=1)
                loss = F.nll_loss(s, y)
            elif config['criterion'] == 'multi-bce':
                y_oh = F.one_hot(y, num_classes=output.size(dim=1))
                s = torch.sigmoid(output)
                loss = F.binary_cross_entropy_with_logits(output, y_oh.float())
            else:
                raise ValueError("config['criterion'] must be set to one of ['cross-entropy', 'multi-bce']")

            # backward and update
            loss.backward()
            optimizer.step()
            scheduler.step()

            # train accuracy
            pred = s.argmax(dim=-1)
            correct += pred.squeeze().eq(y).sum().item()
            total += pred.shape[0]
            cumu_loss += loss.item()

            i += 1
            if steps <= i:
                break
        if steps <= i:
            break

    train_acc = 100.0 * correct / total
    avg_loss = cumu_loss / steps

    return avg_loss, train_acc


def train_mixup_multistep(model, loader, optimizer, scheduler, config, steps):
    model.train()
    device = config['device']

    i = 0
    cumu_loss = 0
    correct, total = (0, 0)

    while True:
        for sample_batched in loader:
            optimizer.zero_grad()

            # load and mixup the mini-batched data
            x1 = sample_batched['signal'].to(device)
            age1 = sample_batched['age'].to(device)
            y1 = sample_batched['class_label'].to(device)

            index = torch.randperm(x1.shape[0]).cuda()
            x2 = x1[index]
            age2 = age1[index]
            y2 = y1[index]

            mixup_alpha = config['mixup']
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            x = lam * x1 + (1.0 - lam) * x2
            age = lam * age1 + (1.0 - lam) * age2

            # forward pass
            output = model(x, age)

            # loss function
            if config['criterion'] == 'cross-entropy':
                s = F.log_softmax(output, dim=1)
                loss1 = F.nll_loss(s, y1)
                loss2 = F.nll_loss(s, y2)
                loss = lam * loss1 + (1 - lam) * loss2
            elif config['criterion'] == 'multi-bce':
                y1_oh = F.one_hot(y1, num_classes=output.size(dim=1))
                y2_oh = F.one_hot(y2, num_classes=output.size(dim=1))
                y_oh = lam * y1_oh + (1.0 - lam) * y2_oh
                s = torch.sigmoid(output)
                loss = F.binary_cross_entropy_with_logits(output, y_oh)
            else:
                raise ValueError("config['criterion'] must be set to one of ['cross-entropy', 'multi-bce']")

            # backward and update
            loss.backward()
            optimizer.step()
            scheduler.step()

            # train accuracy
            pred = s.argmax(dim=-1)
            correct1 = pred.squeeze().eq(y1).sum().item()
            correct2 = pred.squeeze().eq(y2).sum().item()
            correct += lam * correct1 + (1.0 - lam) * correct2
            total += pred.shape[0]
            cumu_loss += loss.item()

            i += 1
            if steps <= i:
                break
        if steps <= i:
            break

    train_acc = 100.0 * correct / total
    avg_loss = cumu_loss / steps

    return avg_loss, train_acc


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


def learning_rate_search(config, train_loader, min_log_lr, max_log_lr, trials, steps):
    learning_rate_record = []
    best_accuracy = 0
    best_model_state = None

    for log_lr in np.linspace(min_log_lr, max_log_lr, num=trials):
        lr = 10 ** log_lr

        model = config['generator'](**config).to(config['device'])
        model.train()

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_decay_step'],
                                              gamma=config['lr_decay_gamma'])

        tr_ms = train_multistep if config.get('mixup', 0) < 1e-12 else train_mixup_multistep
        _, train_accuracy = tr_ms(model, train_loader, optimizer, scheduler, config, steps)

        # Train accuracy for the final epoch is stored
        learning_rate_record.append((log_lr, train_accuracy))

        # keep the best model
        if best_accuracy < train_accuracy:
            best_accuracy = train_accuracy
            best_model_state = deepcopy(model.state_dict())

    best_log_lr = learning_rate_record[np.argmax([v for lr, v in learning_rate_record])][0]

    return 10 ** best_log_lr, learning_rate_record, best_model_state


def draw_loss_plot(losses, lr_decay_step=None):
    plt.style.use('default')  # default, ggplot, fivethirtyeight, classic
    fig = plt.figure(num=1, clear=True, figsize=(8.0, 3.0), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    N = len(losses)
    x = np.arange(1, N + 1)
    ax.plot(x, losses)

    if lr_decay_step is None:
        pass
    elif type(lr_decay_step) is list:
        ax.vlines(lr_decay_step, 0, 1, transform=ax.get_xaxis_transform(),
                  colors='m', alpha=0.5, linestyle='solid')
    else:
        x2 = np.arange(lr_decay_step, N, lr_decay_step)
        ax.vlines(x2, 0, 1, transform=ax.get_xaxis_transform(),
                  colors='m', alpha=0.5, linestyle='solid')
    # ax.vlines([1, N], 0, 1, transform=ax.get_xaxis_transform(),
    #           colors='k', alpha=0.7, linestyle='solid')

    ax.set_xlim(left=0)
    ax.set_title('Loss Plot')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Training Loss')

    plt.show()
    fig.clear()
    plt.close(fig)


def draw_accuracy_history(train_acc_history, val_acc_history, history_interval, lr_decay_step=None):
    plt.style.use('default')  # default, ggplot, fivethirtyeight, classic
    fig = plt.figure(num=1, clear=True, figsize=(8.0, 3.0), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    N = len(train_acc_history) * history_interval
    x = np.arange(history_interval, N + 1, history_interval)
    ax.plot(x, train_acc_history, 'r-', label='Train accuracy')
    ax.plot(x, val_acc_history, 'b-', label='Validation accuracy')

    if lr_decay_step is None:
        pass
    elif type(lr_decay_step) is list:
        ax.vlines(lr_decay_step, 0, 1, transform=ax.get_xaxis_transform(),
                  colors='m', alpha=0.5, linestyle='solid')
    else:
        x2 = np.arange(lr_decay_step, N + 1, lr_decay_step)
        ax.vlines(x2, 0, 1, transform=ax.get_xaxis_transform(),
                  colors='m', alpha=0.5, linestyle='solid')
    # ax.vlines([history_interval, N], 0, 1, transform=ax.get_xaxis_transform(),
    #           colors='k', alpha=0.7, linestyle='solid')

    ax.set_xlim(left=0)
    ax.legend(loc='lower right')
    ax.set_title('Accuracy Plot during Training')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy (%)')

    plt.show()
    fig.clear()
    plt.close(fig)


def draw_confusion(confusion, class_label_to_type, use_wandb=False):
    C = len(class_label_to_type)

    plt.style.use('default')  # default, ggplot, fivethirtyeight, classic
    plt.rcParams['image.cmap'] = 'jet'  # 'nipy_spectral'

    fig = plt.figure(num=1, clear=True, figsize=(4.0, 4.0), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(confusion, alpha=0.8)

    ax.set_xticks(np.arange(C))
    ax.set_yticks(np.arange(C))
    ax.set_xticklabels(class_label_to_type)
    ax.set_yticklabels(class_label_to_type)

    for r in range(C):
        for c in range(C):
            ax.text(c, r, confusion[r, c],
                    ha="center", va="center", color='k')

    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Ground Truth')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # draw
    if use_wandb:
        wandb.log({'Confusion Matrix (Image)': wandb.Image(plt)})
    else:
        plt.show()

    fig.clear()
    plt.close(fig)


def draw_roc_curve(score, target, class_label_to_type, use_wandb=False):
    plt.style.use('default')  # default, ggplot, fivethirtyeight, classic

    # Binarize the output
    n_classes = len(class_label_to_type)
    target = label_binarize(target, classes=np.arange(n_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(target[:, i], score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(target.ravel(), score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # draw class-agnostic ROC curve
    fig = plt.figure(num=1, clear=True, figsize=(8.5, 4.0), constrained_layout=True)
    ax = fig.add_subplot(1, 2, 1)
    lw = 1.5
    colors = cycle(['limegreen', 'mediumpurple', 'darkorange',
                    'dodgerblue', 'lightcoral', 'goldenrod',
                    'indigo', 'darkgreen', 'navy', 'brown'])
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='{0} (area = {1:0.2f})'
                      ''.format(class_label_to_type[i], roc_auc[i]))
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Class-Wise ROC Curves')
    ax.legend(loc="lower right")

    # Plot class-aware ROC curves
    ax = fig.add_subplot(1, 2, 2)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle='-', linewidth=lw)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle='-', linewidth=lw)

    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Class-Agnostic ROC Curves')
    ax.legend(loc="lower right")

    # draw
    if use_wandb:
        wandb.log({'ROC Curve (Image)': wandb.Image(plt)})
    else:
        plt.show()

    fig.clear()
    plt.close(fig)


def draw_debug_table(debug_table, use_wandb=False):
    (debug_table_serial, debug_table_edf, debug_table_pred, debug_table_gt) = debug_table

    fig = plt.figure(num=1, clear=True, figsize=(20.0, 4.0), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    total_error, total_count = (0, 0)

    for edf in np.unique(debug_table_edf):
        indices = [i for i, x in enumerate(debug_table_edf) if x == edf]

        err, cnt = (0, 0)
        for i in indices:
            cnt += sum(debug_table_pred[i])
            err += sum(debug_table_pred[i]) - debug_table_pred[i][debug_table_gt[i]]

        total_error += err
        total_count += cnt

        ax.bar(edf, err / cnt, color=['g', 'b', 'r'][debug_table_gt[indices[0]]])

    ax.set_title(f'Debug Table (Acc. {1.0 - total_error / total_count: .2f}%)', fontsize=18)
    ax.set_ylim(0.0, 1.0)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=9, visible=True)

    if use_wandb:
        table = [[serial, edf, pred, gt] for serial, edf, pred, gt in zip(*debug_table)]
        table = wandb.Table(data=table, columns=['Serial', 'EDF', 'Prediction', 'Ground-truth'])
        wandb.log({'Debug Table': table})

        wandb.log({'Debug Table (Image)': wandb.Image(plt)})
    else:
        plt.show()

    fig.clear()
    plt.close(fig)


def draw_learning_rate_record(learning_rate_record, use_wandb=False):
    if use_wandb:
        data = [[lr, v] for lr, v in learning_rate_record]
        table = wandb.Table(data=data, columns=["learning rate (log)", "train accuracy"])
        wandb.log({"lr_search": wandb.plot.scatter(table, "learning rate (log)", "train accuracy")})
    else:
        plt.style.use('default')  # default, ggplot, fivethirtyeight, classic

        fig = plt.figure(num=1, clear=True, constrained_layout=True, figsize=(5.0, 5.0))
        ax = fig.add_subplot(1, 1, 1)

        ax.set_title('Learning Rate Search')
        ax.set_xlabel('Learning rate in log-scale')
        ax.set_ylabel('Train accuracy')

        ax.scatter(*max(learning_rate_record, key=lambda x: x[1]),
                   s=150, c='w', marker='o', edgecolors='limegreen')

        for log_lr, val_accuracy in learning_rate_record:
            ax.scatter(log_lr, val_accuracy, c='r',
                       alpha=0.5, edgecolors='none')
        plt.show()
        fig.clear()
        plt.close(fig)

