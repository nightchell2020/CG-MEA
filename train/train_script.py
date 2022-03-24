import os
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim
import wandb

from models.utils import count_parameters
from .train_core import train_multistep, train_mixup_multistep
from .evaluate import check_accuracy
from .visualize import draw_learning_rate_record
from .visualize import draw_roc_curve, draw_confusion, draw_debug_table

# __all__ = []


def learning_rate_search(config, train_loader, val_loader,
                         preprocess_train, preprocess_test,
                         trials, steps):
    learning_rate_record = []
    best_accuracy = 0
    best_model_state = None

    # default learning rate range is set based on a minibatch size of 32
    min_log_lr = -2.4 + np.log10(config['minibatch'] / 32)
    max_log_lr = -4.1 + np.log10(config['minibatch'] / 32)

    for log_lr in np.linspace(min_log_lr, max_log_lr, num=trials):
        lr = 10 ** log_lr

        model = config['generator'](**config).to(config['device'])
        model.train()

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=round(config['iterations'] * config['lr_decay_timing']),
                                              gamma=config['lr_decay_gamma'])

        tr_ms = train_multistep if config.get('mixup', 0) < 1e-12 else train_mixup_multistep
        tr_ms(model, train_loader, preprocess_train, optimizer, scheduler, config, steps)

        train_accuracy, *_ = check_accuracy(model, train_loader, preprocess_test, config, 10)
        val_accuracy, *_ = check_accuracy(model, val_loader, preprocess_test, config, 10)

        # Train accuracy for the final epoch is stored
        learning_rate_record.append((log_lr, train_accuracy, val_accuracy))

        # keep the best model
        if best_accuracy < train_accuracy:
            best_accuracy = train_accuracy
            best_model_state = deepcopy(model.state_dict())

    best_log_lr = learning_rate_record[np.argmax([(tr + vl)/2 for _, tr, vl in learning_rate_record])][0]

    return 10 ** best_log_lr, learning_rate_record, best_model_state


def train_with_wandb(config, train_loader, val_loader, test_loader, test_loader_longer,
                     preprocess_train, preprocess_test, class_label_to_type):
    print('*' * 120)
    print(f'{"*" * 30}{config["model"] + " train starts":^60}{"*" * 30}')
    print('*' * 120)

    # search an appropriate starting learning rate if needed
    model_state = None
    if config["LR"] is None:
        config['LR'], lr_search, model_state = learning_rate_search(config=config,
                                                                    train_loader=train_loader,
                                                                    val_loader=val_loader,
                                                                    preprocess_train=preprocess_train,
                                                                    preprocess_test=preprocess_test,
                                                                    trials=25, steps=200)
        wandb.config.LR = config['LR']
        draw_learning_rate_record(lr_search, use_wandb=True)

    # generate model and its trainer
    model = config['generator'](**config).to(config['device'])
    if model_state is not None:
        model.load_state_dict(model_state)

    optimizer = optim.AdamW(model.parameters(),
                            lr=config['LR'],
                            weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=round(config['iterations'] * config['lr_decay_timing']),
                                          gamma=config['lr_decay_gamma'])

    tr_ms = train_multistep if config.get('mixup', 0) < 1e-12 else train_mixup_multistep

    # track gradients and weights statistics if needed
    if config.get('watch_model', None):
        wandb.watch(model, log='all',
                    log_freq=config['history_interval'],
                    log_graph=True)

    # train and validation routine
    best_val_acc = 0
    best_model_state = deepcopy(model.state_dict())
    for i in range(0, config["iterations"], config["history_interval"]):
        # train 'history_interval' steps
        loss, train_acc = tr_ms(model=model,
                                loader=train_loader,
                                preprocess=preprocess_train,
                                optimizer=optimizer,
                                scheduler=scheduler, config=config,
                                steps=config["history_interval"])

        # validation
        val_acc, _, _, _, _ = check_accuracy(model=model,
                                             loader=val_loader,
                                             preprocess=preprocess_test,
                                             config=config, repeat=10)

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_model_state = deepcopy(model.state_dict())
            if config['save_model'] and config['save_temporary']:
                save_path = f'local/checkpoint_temp/{wandb.run.name}/'
                os.makedirs(save_path, exist_ok=True)
                path = os.path.join(save_path, f'{config["model"]}')
                torch.save(best_model_state, path)

        # log
        wandb.log({'Loss': loss,
                   'Train Accuracy': train_acc,
                   'Validation Accuracy': val_acc}, step=i + config["history_interval"])

    # calculate the test accuracies for best and last models
    last_model_state = deepcopy(model.state_dict())
    last_test_result = check_accuracy(model=model,
                                      loader=test_loader,
                                      preprocess=preprocess_test,
                                      config=config, repeat=30)
    last_test_acc = last_test_result[0]

    model.load_state_dict(best_model_state)
    best_test_result = check_accuracy(model=model,
                                      loader=test_loader,
                                      preprocess=preprocess_test,
                                      config=config, repeat=30)
    best_test_acc = best_test_result[0]

    if last_test_acc < best_test_acc:
        model_state = best_model_state
        test_result = best_test_result
    else:
        model_state = last_model_state
        test_result = last_test_result

    model.load_state_dict(model_state)
    test_acc, test_confusion, test_debug, score, target = test_result

    # calculate the test accuracies for final model on much longer sequence
    longer_test_acc = check_accuracy(model=model,
                                     loader=test_loader_longer,
                                     preprocess=preprocess_test,
                                     config=config, repeat=30)[0]

    # save the model
    if config['save_model']:
        save_path = f'local/checkpoint_temp/{wandb.run.name}/'
        os.makedirs(save_path, exist_ok=True)
        path = os.path.join(save_path, f'{config["model"]}')
        torch.save(model_state, path)

    # leave the message
    wandb.config.final_shape = model.get_final_shape()
    wandb.config.num_params = count_parameters(model)
    wandb.log({'Test Accuracy': test_acc,
               '(Best / Last) Test Accuracy': ('Best' if last_test_acc < best_test_acc else 'Last',
                                               round(best_test_acc, 2), round(last_test_acc, 2)),
               'Confusion Matrix (Array)': test_confusion,
               'Test Accuracy (Longer)': longer_test_acc,
               'Test Debug Table/Serial': test_debug[0],
               'Test Debug Table/EDF': test_debug[1],
               'Test Debug Table/Pred': test_debug[2],
               'Test Debug Table/GT': test_debug[3]})

    if config['draw_result']:
        draw_roc_curve(score, target, class_label_to_type, use_wandb=True)
        draw_confusion(test_confusion, class_label_to_type, use_wandb=True)
        draw_debug_table(test_debug, use_wandb=True)
        # leave these disabled to save the wandb resources
        # wandb.log({"Confusion Matrix": wandb.plot.confusion_matrix(y_true=target,
        #                                                            preds=score.argmax(axis=-1),
        #                                                            class_names=class_label_to_type)})
        # wandb.log({"ROC Curve": wandb.plot.roc_curve(target, score, labels=class_label_to_type)})

    return model
