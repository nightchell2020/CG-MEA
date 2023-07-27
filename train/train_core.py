from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from .mixup_util import mixup_data, mixup_criterion
from .evaluate import compute_feature_embedding

# __all__ = []


def train_multistep(model, loader, preprocess, optimizer, scheduler, amp_scaler, config, steps):
    model.train()

    i = 0
    cumu_loss = 0
    correct, total = (0, 0)

    while True:
        for sample_batched in loader:
            optimizer.zero_grad()

            # keep the data for knowledge distillation
            if config.get("distil_teacher", None):
                sample_batched_distil = deepcopy(sample_batched)

            # preprocessing (this includes to-device operation)
            preprocess(sample_batched)

            # pull the data
            x = sample_batched["signal"]
            age = sample_batched["age"]
            y = sample_batched["class_label"]

            # mixup the mini-batched data
            x, age, y1, y2, lam, mixup_index = mixup_data(x, age, y, config["mixup"])

            # mixed precision training if needed
            with autocast(enabled=config.get("mixed_precision", False)):
                # forward pass
                output = model(x, age)
                if isinstance(output, tuple):
                    output, output_kd = output
                else:
                    output_kd = output

                if config["use_age"] == "estimate":
                    output_age = output[:, -1]
                    output = output[:, :-1]

                # loss function
                if config["criterion"] == "cross-entropy":
                    loss = mixup_criterion(F.cross_entropy, output, y1, y2, lam)
                elif config["criterion"] == "multi-bce":
                    y1_oh = F.one_hot(y1, num_classes=output.size(dim=1))
                    y2_oh = F.one_hot(y2, num_classes=output.size(dim=1))
                    loss = mixup_criterion(
                        F.binary_cross_entropy_with_logits,
                        output,
                        y1_oh.float(),
                        y2_oh.float(),
                        lam,
                    )
                elif config["criterion"] == "svm":
                    loss = mixup_criterion(F.multi_margin_loss, output, y1, y2, lam)
                else:
                    raise ValueError("config['criterion'] must be set to one of ['cross-entropy', 'multi-bce', 'svm']")

                # distillation loss computation
                if config.get("distil_teacher", None):
                    if config.get("distil_teacher_model", None):
                        output_teacher = compute_feature_embedding(
                            config["distil_teacher_model"],
                            sample_batched_distil,
                            config["distil_teacher_preprocess"],
                            config,
                            target_from_last=0,
                        )
                    elif "distil_teacher_score" in config.keys():
                        output_teacher = config["distil_teacher_score"][
                            torch.tensor(list(map(int, sample_batched["serial"])), dtype=torch.long),
                            sample_batched["crop_timing"],
                        ]
                        # output_teacher = config["distil_teacher_score"][[*list(map(int, sample_batched["serial"]))]]
                    else:
                        raise ValueError(
                            "Any of config['distil_teacher_model'] and config['distil_teacher_score'] is set."
                        )

                    distil_tau = config.get("distil_tau", 1.0)

                    if config["criterion"] == "cross-entropy":
                        if config.get("distil_type") == "hard":
                            output_teacher = output_teacher.argmax(dim=1)
                            distil_loss = mixup_criterion(
                                F.cross_entropy, output_kd, output_teacher, output_teacher[mixup_index], lam
                            )
                        elif config.get("distil_type") == "soft":
                            output_teacher = F.log_softmax(output_teacher / distil_tau, dim=1)
                            distil_loss = mixup_criterion(
                                F.kl_div,
                                F.log_softmax(output_kd / distil_tau, dim=1),
                                output_teacher,
                                output_teacher[mixup_index],
                                lam,
                                reduction="sum",
                                log_target=True,
                            )
                            distil_loss = distil_loss * (distil_tau * distil_tau) / output_kd.numel()

                    elif config["criterion"] == "multi-bce":
                        if config.get("distil_type") == "hard":
                            teacher_y_oh = F.one_hot(
                                output_teacher.argmax(dim=1),
                                num_classes=output_kd.size(dim=1),
                            )
                            loss = mixup_criterion(
                                F.binary_cross_entropy_with_logits,
                                output_kd,
                                teacher_y_oh.float(),
                                teacher_y_oh[mixup_index].float(),
                                lam,
                            )
                        elif config.get("distil_type") == "soft":
                            output_teacher = (output_teacher / distil_tau).sigmoid()
                            distil_loss = mixup_criterion(
                                F.binary_cross_entropy_with_logits,
                                output_kd / distil_tau,
                                output_teacher,
                                output_teacher[mixup_index],
                                lam,
                                reduction="sum",
                            )
                            distil_loss = distil_loss * (distil_tau * distil_tau) / output_kd.numel()

                    elif config["criterion"] == "svm":
                        if config.get("distil_type") == "hard":
                            output_teacher = output_teacher.argmax(dim=1)
                            distil_loss = mixup_criterion(
                                F.multi_margin_loss,
                                output_kd,
                                output_teacher,
                                output_teacher[mixup_index],
                                lam,
                            )

                    distil_alpha = config["distil_alpha"]
                    loss = (1 - distil_alpha) * loss + distil_alpha * distil_loss

            # backward and update
            if config.get("mixed_precision", False):
                amp_scaler.scale(loss).backward()
                if "clip_grad_norm" in config:
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
                amp_scaler.step(optimizer)
                amp_scaler.update()
                scheduler.step()
            else:
                loss.backward()
                if "clip_grad_norm" in config:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
                optimizer.step()
                scheduler.step()

            # train accuracy
            pred = output.argmax(dim=-1)
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


def ssl_train_multistep(model, loader, preprocess, optimizer, scheduler, amp_scaler, config, steps):
    model.train()

    i = 0
    cumu_loss = 0

    while True:
        for sample_batched in loader:
            optimizer.zero_grad()

            # preprocessing (this includes to-device operation)
            preprocess(sample_batched)

            # pull the data
            x = sample_batched["signal"]
            age = sample_batched["age"]

            # mixed precision training if needed
            with autocast(enabled=config.get("mixed_precision", False)):
                # forward pass
                loss = model(x, age)

            # backward and update
            if config.get("mixed_precision", False):
                amp_scaler.scale(loss).backward()
                if "clip_grad_norm" in config:
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
                amp_scaler.step(optimizer)
                amp_scaler.update()
                scheduler.step()
            else:
                loss.backward()
                if "clip_grad_norm" in config:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
                optimizer.step()
                scheduler.step()

            # post update (e.g., momentum)
            if config.get("ddp", False):
                model.module.post_update_params()
            else:
                model.post_update_params()

            # train accuracy
            cumu_loss += loss.item()

            i += 1
            if steps <= i:
                break
        if steps <= i:
            break

    avg_loss = cumu_loss / steps

    return avg_loss
