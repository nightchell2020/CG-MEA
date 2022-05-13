import os
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim
import wandb
import pprint
from datetime import datetime

from .train_core import train_multistep, train_mixup_multistep
from optim import get_lr_scheduler
from .evaluate import check_accuracy
from .evaluate import check_accuracy_extended
from .evaluate import check_accuracy_multicrop
from .visualize import draw_learning_rate_record
from .visualize import draw_roc_curve, draw_confusion, draw_error_table

# __all__ = []


def learning_rate_search(config, model, train_loader, val_loader,
                         preprocess_train, preprocess_test,
                         trials, steps):
    learning_rate_record = []
    best_accuracy = 0
    best_model_state = None

    # default learning rate range is set based on a minibatch size of 32
    min_log_lr = -3.0 + np.log10(config['minibatch'] / 32)
    max_log_lr = -6.0 + np.log10(config['minibatch'] / 32)

    for log_lr in np.linspace(min_log_lr, max_log_lr, num=trials):
        lr = 10 ** log_lr

        if config['ddp']:
            model.module.reset_weights()
        else:
            model.reset_weights()
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

        del optimizer, scheduler
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    model.load_state_dict(best_model_state)
    best_log_lr = learning_rate_record[np.argmax([(tr + vl)/2 for _, tr, vl in learning_rate_record])][0]

    return 10 ** best_log_lr, learning_rate_record


def train_script(config, model, train_loader, val_loader, test_loader, multicrop_test_loader,
                 preprocess_train, preprocess_test):
    # only the main process of DDP logs, evaluates, and saves
    main_process = config['ddp'] is False or config['device'] == 0
    if main_process and config['use_wandb']:
        wandb.init(project=config.get('project', 'noname'), reinit=True)
        wandb.run.name = wandb.run.id

    # training iteration and other conditions
    config['iterations'] = round(config['total_samples'] / config['minibatch'] / config.get('ddp_size', 1))
    history_interval = max(config['iterations'] // config['num_history'], 1)
    config['warmup_steps'] = max(round(config['iterations'] * config['warmup_ratio']), config['warmup_min'])

    # search an appropriate starting learning rate if needed
    if config['search_lr']:
        config['base_lr'], lr_search = learning_rate_search(config=config, model=model,
                                                            train_loader=train_loader, val_loader=val_loader,
                                                            preprocess_train=preprocess_train,
                                                            preprocess_test=preprocess_test,
                                                            trials=25, steps=300)

        # model.reset_weights()  # This line can be remained or commented out.

        if main_process:
            draw_learning_rate_record(lr_search, use_wandb=config['use_wandb'])

    # generate the trainers
    config['base_lr'] = config['base_lr'] * config.get('search_multiplier', 1.0)

    optimizer = optim.AdamW(model.parameters(), lr=config['base_lr'],
                            weight_decay=config['weight_decay'])

    scheduler = get_lr_scheduler(optimizer, config['lr_scheduler_type'],
                                 iterations=config['iterations'],
                                 warmup_steps=config['warmup_steps'])

    tr_ms = train_multistep if config.get('mixup', 0) < 1e-12 else train_mixup_multistep

    if main_process:
        # update configurations
        if config['use_wandb']:
            wandb.config.update(config)

        # track gradients and weights statistics if needed
        if config['use_wandb'] and config.get('watch_model', False):
            wandb.watch(model, log='all', log_freq=history_interval, log_graph=True)

        # directory to save
        run_name = wandb.run.name if config['use_wandb'] else datetime.now().strftime("%Y_%m%d_%H%M")

        if config['save_model']:
            save_path = f'local/checkpoint_temp/{run_name}/'
            if 'cwd' in config:
                save_path = os.path.join(config['cwd'], save_path)
            os.makedirs(save_path, exist_ok=True)

    # train and validation routine
    best_val_acc = 0
    best_model_state = deepcopy(model.state_dict())

    for i in range(0, config["iterations"], history_interval):
        if config.get('ddp', False):
            torch.distributed.barrier()

        # train for 'history_interval' steps
        loss, train_acc = tr_ms(model=model,
                                loader=train_loader,
                                preprocess=preprocess_train,
                                optimizer=optimizer,
                                scheduler=scheduler, config=config,
                                steps=history_interval)
        # validation
        val_acc = check_accuracy(model=model, loader=val_loader,
                                 preprocess=preprocess_test,
                                 config=config, repeat=30)
        # log
        if main_process:
            if config['use_wandb']:
                wandb.log({'Loss': loss,
                           'Train Accuracy': train_acc,
                           'Validation Accuracy': val_acc,
                           'Learning Rate': optimizer.state_dict()['param_groups'][0]['lr'],
                           }, step=(i + history_interval) * config["minibatch"])
            else:
                print(f"{i:7>} / {config['iterations']:>7} iter - "
                      f"Loss: {loss:.4}, Train Acc.: {train_acc:.4}, Val. Acc.: {val_acc:.4}")

            # save the best model so far
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_model_state = deepcopy(model.state_dict())
                if config['save_model']:
                    best_checkpoint = {'model_state': best_model_state, 'config': config,
                                       'optimizer_state': optimizer.state_dict(),
                                       'scheduler_state': scheduler.state_dict()}
                    torch.save(best_checkpoint, os.path.join(save_path, 'best_checkpoint.pt'))

    # calculate the test accuracy for best and last models
    if main_process:
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

        # calculate the test accuracy of the final model using multiple crop averaging
        multicrop_test_acc = check_accuracy_multicrop(model=model, loader=multicrop_test_loader,
                                                      preprocess=preprocess_test,
                                                      config=config, repeat=30)

        # save the model
        if config['save_model']:
            last_checkpoint = {'model_state': last_model_state, 'config': config,
                               'optimizer_state': optimizer.state_dict(),
                               'scheduler_state': scheduler.state_dict()}
            torch.save(last_checkpoint, os.path.join(save_path, 'last_checkpoint.pt'))

        # leave the message
        if config['use_wandb']:
            wandb.log({'Test Accuracy': test_acc,
                       '(Best, Last) Test Accuracy': ('Best' if last_test_acc < best_test_acc else 'Last',
                                                      round(best_test_acc, 2), round(last_test_acc, 2)),
                       'Confusion Matrix (Array)': test_confusion,
                       'Multi-Crop Test Accuracy': multicrop_test_acc,
                       'Test Debug Table/Serial': error_table['Serial'],
                       'Test Debug Table/Pred': error_table['Pred'],
                       'Test Debug Table/GT': error_table['GT']})
        else:
            print(f"\n{'*'*30} {run_name:^30} {'*'*30}\n")
            pprint.pprint({'Test Accuracy': test_acc,
                           '(Best, Last) Test Accuracy': ('Best' if last_test_acc < best_test_acc else 'Last',
                                                          round(best_test_acc, 2), round(last_test_acc, 2)),
                           'Confusion Matrix (Array)': test_confusion,
                           'Multi-Crop Test Accuracy': multicrop_test_acc})
            print(f"\n{'*'*92}\n")

        if config['draw_result']:
            draw_roc_curve(score, target, config['class_label_to_name'], use_wandb=config['use_wandb'])
            draw_confusion(test_confusion, config['class_label_to_name'], use_wandb=config['use_wandb'])
            draw_error_table(error_table, use_wandb=config['use_wandb'])

        if config['use_wandb']:
            wandb.run.finish()

        del last_model_state

    # release memory
    del optimizer, scheduler, best_model_state
    return
