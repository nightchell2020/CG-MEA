import os
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim
import wandb

from models.utils import count_parameters
from .train_core import train_multistep, train_mixup_multistep
from optim import get_lr_scheduler
from .evaluate import check_accuracy, check_accuracy_extended
from .visualize import draw_learning_rate_record
from .visualize import draw_roc_curve, draw_confusion, draw_error_table

# __all__ = []


def learning_rate_search(config, train_loader, val_loader,
                         preprocess_train, preprocess_test,
                         trials, steps):
    learning_rate_record = []
    best_accuracy = 0
    best_model_state = None

    # default learning rate range is set based on a minibatch size of 32
    min_log_lr = -2.4 + np.log10(config['minibatch'] / 32)
    max_log_lr = -4.3 + np.log10(config['minibatch'] / 32)

    for log_lr in np.linspace(min_log_lr, max_log_lr, num=trials):
        lr = 10 ** log_lr

        model = config['generator'](**config).to(config['device'])
        model.train()

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
        scheduler = get_lr_scheduler(optimizer, scheduler_type='constant_with_decay',  # constant for search
                                     iterations=config['iterations'], warmup_steps=config['iterations'])

        tr_ms = train_multistep if config.get('mixup', 0) < 1e-12 else train_mixup_multistep
        tr_ms(model, train_loader, preprocess_train, optimizer, scheduler, config, steps)

        train_accuracy = check_accuracy(model, train_loader, preprocess_test, config, 10)
        val_accuracy = check_accuracy(model, val_loader, preprocess_test, config, 10)

        # Train accuracy for the final epoch is stored
        learning_rate_record.append((log_lr, train_accuracy, val_accuracy))

        # keep the best model
        if best_accuracy < (train_accuracy + val_accuracy) / 2:
            best_accuracy = (train_accuracy + val_accuracy) / 2
            best_model_state = deepcopy(model.state_dict())

        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    best_log_lr = learning_rate_record[np.argmax([(tr + vl)/2 for _, tr, vl in learning_rate_record])][0]

    return 10 ** best_log_lr, learning_rate_record, best_model_state


def train_with_wandb(config, train_loader, val_loader, test_loader, test_loader_longer,
                     preprocess_train, preprocess_test):
    print('*' * 120)
    print(f'{"*" * 30}{config["model"] + " train starts":^60}{"*" * 30}')
    print('*' * 120)

    # search an appropriate starting learning rate if needed
    model_state = None
    if config.get('LR', None) is None:
        config['LR'], lr_search, model_state = learning_rate_search(config=config,
                                                                    train_loader=train_loader,
                                                                    val_loader=val_loader,
                                                                    preprocess_train=preprocess_train,
                                                                    preprocess_test=preprocess_test,
                                                                    trials=30,
                                                                    steps=150)
        draw_learning_rate_record(lr_search, use_wandb=True)

    # generate model and its trainer
    model = config['generator'](**config).to(config['device'])

    config['output_length'] = model.get_output_length()
    config['num_params'] = count_parameters(model)
    for k, v in config.items():
        if k not in wandb.config or wandb.config[k] is None:
            wandb.config[k] = v

    # if model_state is not None:
    #     model.load_state_dict(model_state)

    optimizer = optim.AdamW(model.parameters(),
                            lr=config['LR'] * config.get('search_multiplier', 1.0),
                            weight_decay=config['weight_decay'])

    scheduler = get_lr_scheduler(optimizer,
                                 config['lr_scheduler_type'],
                                 iterations=config['iterations'],
                                 warmup_steps=config.get('warmup_steps', 10000))

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
        val_acc = check_accuracy(model=model, loader=val_loader,
                                 preprocess=preprocess_test,
                                 config=config, repeat=10)

        wandb.log({'Loss': loss, 'Train Accuracy': train_acc, 'Validation Accuracy': val_acc},
                  step=(i + config["history_interval"]) * config["minibatch"])

        # save the best model so far
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_model_state = deepcopy(model.state_dict())
            if config['save_model']:
                save_path = f'local/checkpoint_temp/{wandb.run.name}/'
                os.makedirs(save_path, exist_ok=True)
                best_checkpoint = {'model_state': best_model_state,
                                   'config': config,
                                   'optimizer_state': optimizer.state_dict(),
                                   'scheduler_state': scheduler.state_dict()}
                torch.save(best_checkpoint, os.path.join(save_path, 'best_checkpoint.pt'))

    # calculate the test accuracies for best and last models
    last_model_state = deepcopy(model.state_dict())
    last_test_result = check_accuracy_extended(model=model, loader=test_loader,
                                               preprocess=preprocess_test,
                                               config=config, repeat=30)
    last_test_acc = last_test_result[0]

    model.load_state_dict(best_model_state)
    best_test_result = check_accuracy_extended(model=model, loader=test_loader,
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
    test_acc, score, target, test_confusion, error_table = test_result

    # calculate the test accuracy for final model on much longer sequence
    longer_test_acc = check_accuracy(model=model, loader=test_loader_longer,
                                     preprocess=preprocess_test,
                                     config=config, repeat=30)

    # save the model
    if config['save_model']:
        save_path = f'local/checkpoint_temp/{wandb.run.name}/'
        os.makedirs(save_path, exist_ok=True)
        last_checkpoint = {'model_state': last_model_state,
                           'config': config,
                           'optimizer_state': optimizer.state_dict(),
                           'scheduler_state': scheduler.state_dict()}
        torch.save(last_checkpoint, os.path.join(save_path, 'last_checkpoint.pt'))

    # leave the message
    wandb.log({'Test Accuracy': test_acc,
               '(Best / Last) Test Accuracy': ('Best' if last_test_acc < best_test_acc else 'Last',
                                               round(best_test_acc, 2), round(last_test_acc, 2)),
               'Confusion Matrix (Array)': test_confusion,
               'Test Accuracy (Longer)': longer_test_acc,
               'Test Debug Table/Serial': error_table['Serial'],
               'Test Debug Table/Pred': error_table['Pred'],
               'Test Debug Table/GT': error_table['GT']})

    if config['draw_result']:
        draw_roc_curve(score, target, config['class_label_to_name'], use_wandb=True)
        draw_confusion(test_confusion, config['class_label_to_name'], use_wandb=True)
        draw_error_table(error_table, use_wandb=True)

        # leave these disabled to save the wandb resources
        # wandb.log({"Confusion Matrix": wandb.plot.confusion_matrix(y_true=target,
        #                                                            preds=score.argmax(axis=-1),
        #                                                            class_names=config['class_label_to_name'])})
        # wandb.log({"ROC Curve": wandb.plot.roc_curve(target, score, labels=config['class_label_to_name'])})

    # release memory
    del optimizer, scheduler
    del last_model_state, best_model_state

    return model
