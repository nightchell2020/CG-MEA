from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


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
