from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  # roc_auc_score
from sklearn.preprocessing import label_binarize
import wandb


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

