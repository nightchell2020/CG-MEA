import os
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim
import wandb
import pprint
from datetime import datetime

from .train_core import ssl_train_multistep
from optim import get_lr_scheduler
from .visualize import draw_lr_search_record

# __all__ = []


def learning_rate_search(config, model, loader, preprocess, trials, steps):
    learning_rate_record = []
    given_model_state = deepcopy(model.state_dict())

    # default learning rate range is set based on a minibatch size of 32
    min_log_lr = -3.2 + np.log10(config['minibatch'] * config.get('ddp_size', 1) / 32)
    max_log_lr = -6.0 + np.log10(config['minibatch'] * config.get('ddp_size', 1) / 32)

    for log_lr in np.linspace(min_log_lr, max_log_lr, num=trials):
        lr = 10 ** log_lr

        # recover the given  model state
        model.load_state_dict(deepcopy(given_model_state))
        # model.module.reset_weights() if config.get('ddp', False) else model.reset_weights()

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
        scheduler = get_lr_scheduler(optimizer, scheduler_type='constant_with_decay',  # constant for search
                                     iterations=config['total_samples'], warmup_steps=config['total_samples'])
        amp_scaler = torch.cuda.amp.GradScaler() if config.get('mixed_precision', False) else None

        loss = ssl_train_multistep(model, loader, preprocess, optimizer, scheduler, amp_scaler, config, steps)

        # Train accuracy for the final epoch is stored
        learning_rate_record.append((log_lr, loss))

        del optimizer, scheduler
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # find the best starting point (if a tie occurs, average them)
    losses = np.array([loss for _, loss in learning_rate_record])
    induces = np.argwhere(losses == np.max(losses))
    best_log_lr = np.average(np.array([log_lr for log_lr, _ in learning_rate_record])[induces])

    # recover the given  model state
    model.load_state_dict(deepcopy(given_model_state))

    return 10 ** best_log_lr, learning_rate_record


def ssl_train_script(config, model, loader, preprocess):
    # only the main process of DDP logs, evaluates, and saves
    main_process = config['ddp'] is False or config['device'].index == 0

    if main_process:
        print(f"\n{'*'*30} {'Configurations for Train':^30} {'*'*30}\n")
        pprint.pprint(config, width=120)
        print(f"\n{'*'*92}\n")

    # load if using an existing model
    if config.get('init_from', None):
        init_path = os.path.join(config.get('cwd', ''), f'local/checkpoint/{config["init_from"]}/')
        checkpoint = torch.load(os.path.join(init_path, 'checkpoint.pt'), map_location=config['device'])
        model.load_state_dict(checkpoint['ssl_model_state'])
        pprint.pprint(f'Load an existing model from {config["init_from"]}\n', width=120)

    # wandb init
    if main_process and config['use_wandb']:
        if config.get('resume', None) is None:
            wandb.init(project=config.get('project', 'noname'), reinit=True)
            wandb.run.name = wandb.run.id
        else:
            wandb.init(project=config.get('project', 'noname'), id=config["resume"], resume='must')

    # search an appropriate starting learning rate if needed
    if config.get('search_lr', False) and config.get('resume', None) is None:
        config['base_lr'], lr_search = learning_rate_search(config=config, model=model,
                                                            loader=loader, preprocess=preprocess,
                                                            trials=20, steps=500)
        if main_process:
            draw_lr_search_record(lr_search, use_wandb=config['use_wandb'])

    # training iteration and other conditions
    config['base_lr'] = config['base_lr'] * config.get('search_multiplier', 1.0)
    config['iterations'] = round(config['total_samples'] / config['minibatch'] / config.get('ddp_size', 1))
    config['warmup_steps'] = max(round(config['iterations'] * config['warmup_ratio']), config['warmup_min'])
    history_interval = max(config['iterations'] // config['num_history'], 1)

    # generate the trainers
    optimizer = optim.AdamW(model.parameters(), lr=config['base_lr'], weight_decay=config['weight_decay'])
    scheduler = get_lr_scheduler(optimizer, config['lr_scheduler_type'],
                                 iterations=config['iterations'], warmup_steps=config['warmup_steps'])
    amp_scaler = torch.cuda.amp.GradScaler() if config.get('mixed_precision', False) else None

    # local variable for training loop
    i_step = 0

    # load if resuming
    if config.get('resume', None):
        resume = config['resume']
        save_path = os.path.join(config.get('cwd', ''), f'local/checkpoint/{config["resume"]}/')
        checkpoint = torch.load(os.path.join(save_path, 'checkpoint.pt'), map_location=config['device'])
        model.load_state_dict(checkpoint['ssl_model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        config = checkpoint['config']
        if main_process and config['use_wandb']:
            wandb.config.update(config, allow_val_change=True)
        i_step = checkpoint['optimizer_state']['state'][0]['step']
        if not isinstance(i_step, int):
            i_step = int(i_step.detach().cpu().numpy())
        print(f"\n{'*'*30} {f'Training resumes from {resume}':^30} {'*'*30}\n")
        pprint.pprint(config, width=120)
        print(f"\n{'*'*92}\n")

    if main_process:
        # update configurations
        if config['use_wandb']:
            wandb.config.update(config)

            # track gradients and weights statistics if needed
            if config.get('watch_model', False):
                wandb.watch(model, log='all', log_freq=history_interval, log_graph=True)

        # directory to save
        run_name = wandb.run.name if config['use_wandb'] else datetime.now().strftime("%Y-%m%d-%H%M")
        if config['save_model']:
            save_path = os.path.join(config.get('cwd', ''), f'local/checkpoint/{run_name}/')
            os.makedirs(save_path, exist_ok=True)

    # train and validation routine
    while i_step < config["iterations"]:
        i_step += history_interval

        # train during 'history_interval' steps
        loss = ssl_train_multistep(model=model, loader=loader, preprocess=preprocess,
                                   optimizer=optimizer, scheduler=scheduler, amp_scaler=amp_scaler,
                                   config=config, steps=history_interval)
        # log
        if main_process:
            if config['use_wandb']:
                wandb.log({'Loss': loss, 'Learning Rate': optimizer.state_dict()['param_groups'][0]['lr'], },
                          step=i_step * config["minibatch"])
            else:
                print(f"{i_step:7>} / {config['iterations']:>7} iter - Loss: {loss:.4}")

            # save the model
            if config['save_model']:
                checkpoint = {'model_state': model.backbone.state_dict(), 'ssl_model_state': model.state_dict(),
                              'config': config, 'optimizer_state': optimizer.state_dict(),
                              'scheduler_state': scheduler.state_dict()}
                torch.save(checkpoint, os.path.join(save_path, 'checkpoint_.pt'))
                os.replace(os.path.join(save_path, 'checkpoint_.pt'), os.path.join(save_path, 'checkpoint.pt'))

    # calculate the test accuracy for best and last models
    if main_process:
        # save the model
        if config['save_model']:
            checkpoint = {'model_state': model.backbone.state_dict(), 'ssl_model_state':  model.state_dict(),
                          'config': config, 'optimizer_state': optimizer.state_dict(),
                          'scheduler_state': scheduler.state_dict()}
            torch.save(checkpoint, os.path.join(save_path, 'checkpoint.pt'))
            os.replace(os.path.join(save_path, 'checkpoint.pt'), os.path.join(save_path, 'checkpoint.pt'))

        # leave the message
        if config['use_wandb']:
            wandb.run.finish()

    # release memory
    del optimizer, scheduler
    return
