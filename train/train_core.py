import numpy as np
import torch
import torch.nn.functional as F

from .utils import TimeElapsed

# __all__ = []


def train_multistep(model, loader, preprocess, optimizer, scheduler, config, steps):
    model.train()

    i = 0
    cumu_loss = 0
    correct, total = (0, 0)

    while True:
        te = TimeElapsed()
        for sample_batched in loader:
            print(te.elapsed_str())

            optimizer.zero_grad()
            print(te.elapsed_str())

            # preprocessing (this includes to-device operation)
            preprocess(sample_batched)
            print(te.elapsed_str())

            # forward pass
            x = sample_batched['signal']
            age = sample_batched['age']
            y = sample_batched['class_label']
            output = model(x, age)
            print(te.elapsed_str())

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
            print(te.elapsed_str())

            # backward and update
            loss.backward()
            print(te.elapsed_str())
            optimizer.step()
            print(te.elapsed_str())
            scheduler.step()
            print(te.elapsed_str())

            # train accuracy
            pred = s.argmax(dim=-1)
            correct += pred.squeeze().eq(y).sum().item()
            total += pred.shape[0]
            cumu_loss += loss.item()
            print(te.elapsed_str())
            print()

            i += 1
            if steps <= i:
                break
        print(te.elapsed_str())
        print('-' * 70)
        if steps <= i:
            break

    train_acc = 100.0 * correct / total
    avg_loss = cumu_loss / steps

    return avg_loss, train_acc


def train_mixup_multistep(model, loader, preprocess, optimizer, scheduler, config, steps):
    model.train()

    i = 0
    cumu_loss = 0
    correct, total = (0, 0)

    while True:
        for sample_batched in loader:
            optimizer.zero_grad()

            # preprocessing (this includes to-device operation)
            preprocess(sample_batched)

            # load and mixup the mini-batched data
            x1 = sample_batched['signal']
            age1 = sample_batched['age']
            y1 = sample_batched['class_label']

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
