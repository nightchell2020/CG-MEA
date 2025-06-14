import numpy as np
import torch
import torch.nn.functional as F

# __all__ = []


@torch.no_grad()
def compute_feature_embedding(model, sample_batched, preprocess, config, target_from_last=1):
    # evaluation mode
    model.eval()

    # preprocessing (this includes to-device operation)
    preprocess(sample_batched)

    # apply model on whole batch directly on device
    x = sample_batched["signal"]
    age = sample_batched["age"]

    module = model.module if config.get("ddp", False) else model
    output = module.compute_feature_embedding(x, age, target_from_last=target_from_last)

    # DeiT model
    if isinstance(output, tuple):
        output = (output[0] + output[1]) / 2.0

    return output


@torch.no_grad()
def estimate_logit(model, sample_batched, preprocess, config):
    output = compute_feature_embedding(model, sample_batched, preprocess, config, target_from_last=0)
    return output


@torch.no_grad()
def logit_to_prob(logit, config):
    # map depending on the model's loss function
    if config["criterion"] == "cross-entropy":
        score = F.softmax(logit, dim=1)
    elif config["criterion"] == "multi-bce":
        score = torch.sigmoid(logit)
    elif config["criterion"] == "svm":
        score = logit
    else:
        raise ValueError(f"logit_to_prob(): cannot parse config['criterion']={config['criterion']}.")
    return score


@torch.no_grad()
def estimate_class_score(model, sample_batched, preprocess, config):
    output = compute_feature_embedding(model, sample_batched, preprocess, config, target_from_last=0)
    output = logit_to_prob(output, config)
    return output


def calculate_confusion_matrix(pred, target, num_classes):
    N = target.shape[0]
    C = num_classes
    confusion = np.zeros((C, C), dtype=np.int32)

    for i in range(N):
        r = target[i]
        c = pred[i]
        confusion[r, c] += 1
    return confusion


def calculate_confusion_matrix2(pred, target, num_classes, set_size):
    N = set_size
    repeat = target.shape[0] // N
    confusion = []
    for r in range(repeat):
        cm = calculate_confusion_matrix(
            pred[r * N : (r + 1) * N],
            target[r * N : (r + 1) * N],
            num_classes=num_classes,
        )
        confusion.append(cm)
    return np.array(confusion)


def calculate_class_wise_metrics(confusion_matrix):
    n_classes = confusion_matrix.shape[0]

    accuracy = np.zeros((n_classes,))
    sensitivity = np.zeros((n_classes,))
    specificity = np.zeros((n_classes,))
    precision = np.zeros((n_classes,))
    recall = np.zeros((n_classes,))

    for c in range(n_classes):
        tp = confusion_matrix[c, c]
        fn = confusion_matrix[c].sum() - tp
        fp = confusion_matrix[:, c].sum() - tp
        tn = confusion_matrix.sum() - tp - fn - fp

        accuracy[c] = (tp + tn) / (tp + fn + fp + tn)
        sensitivity[c] = tp / (tp + fn)
        specificity[c] = tn / (fp + tn)
        precision[c] = tp / (tp + fp)
        recall[c] = sensitivity[c]
    f1_score = 2 * precision * recall / (precision + recall)

    class_wise_metrics = {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1-score": f1_score,
    }  # 'Recall': recall is same with sensitivity
    return class_wise_metrics


@torch.no_grad()
def check_accuracy(model, loader, preprocess, config, repeat=1):
    # for accuracy
    correct, total = (0, 0)

    for k in range(repeat):
        for sample_batched in loader:
            # estimate
            s = estimate_class_score(model, sample_batched, preprocess, config)
            y = sample_batched["class_label"]

            # calculate accuracy
            pred = s.argmax(dim=-1)
            correct += pred.squeeze().eq(y).sum().item()
            total += pred.shape[0]

    accuracy = 100.0 * correct / total
    return accuracy


@torch.no_grad()
def check_accuracy_extended(model, loader, preprocess, config, repeat=1, dummy=1):
    # for confusion matrix
    C = config["out_dims"]
    confusion_matrix = np.zeros((C, C), dtype=np.int32)

    # for ROC curve
    score = None
    target = None

    # for throughput calculation
    total = 0
    total_time = 0.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # warm-up using dummy round
    for k in range(dummy):
        for sample_batched in loader:
            _ = estimate_class_score(model, sample_batched, preprocess, config)

    for k in range(repeat):
        for sample_batched in loader:
            # estimate
            start_event.record()
            s = estimate_class_score(model, sample_batched, preprocess, config)
            end_event.record()
            torch.cuda.synchronize()
            total_time += start_event.elapsed_time(end_event) / 1000

            y = sample_batched["class_label"]

            # classification score for drawing ROC curve
            if score is None:
                score = s.detach().cpu().numpy()
                target = y.detach().cpu().numpy()
            else:
                score = np.concatenate((score, s.detach().cpu().numpy()), axis=0)
                target = np.concatenate((target, y.detach().cpu().numpy()), axis=0)

            # confusion matrix
            pred = s.argmax(dim=-1)
            confusion_matrix += calculate_confusion_matrix(pred, y, num_classes=config["out_dims"])

            # total samples
            total += pred.shape[0]

    accuracy = confusion_matrix.trace() / confusion_matrix.sum() * 100.0
    throughput = total / total_time

    return accuracy, score, target, confusion_matrix, throughput


@torch.no_grad()
def check_accuracy_extended_debug(model, loader, preprocess, config, repeat=1):
    # for confusion matrix
    C = config["out_dims"]
    confusion_matrix = np.zeros((C, C), dtype=np.int32)

    # for error table
    error_table = {data["serial"]: {"GT": data["class_label"].item(), "Pred": [0] * C} for data in loader.dataset}

    # for crop timing
    crop_timing = dict()

    # for ROC curve
    score = None
    target = None

    for k in range(repeat):
        for sample_batched in loader:
            # estimate
            s = estimate_class_score(model, sample_batched, preprocess, config)
            y = sample_batched["class_label"]

            # classification score for drawing ROC curve
            if score is None:
                score = s.detach().cpu().numpy()
                target = y.detach().cpu().numpy()
            else:
                score = np.concatenate((score, s.detach().cpu().numpy()), axis=0)
                target = np.concatenate((target, y.detach().cpu().numpy()), axis=0)

            # confusion matrix
            pred = s.argmax(dim=-1)
            confusion_matrix += calculate_confusion_matrix(pred, y, num_classes=config["out_dims"])

            # error table
            for n in range(pred.shape[0]):
                serial = sample_batched["serial"][n]
                error_table[serial]["Pred"][pred[n].item()] += 1

            # crop timing
            for n in range(pred.shape[0]):
                ct = sample_batched["crop_timing"][n]

                if ct not in crop_timing.keys():
                    crop_timing[ct] = {}

                if pred[n] == y[n]:
                    crop_timing[ct]["correct"] = crop_timing[ct].get("correct", 0) + 1
                else:
                    crop_timing[ct]["incorrect"] = crop_timing[ct].get("incorrect", 0) + 1

    # error table update
    error_table_serial = []
    error_table_pred = []
    error_table_gt = []

    for serial in sorted(error_table.keys()):
        error_table_serial.append(serial)
        error_table_pred.append(error_table[serial]["Pred"])
        error_table_gt.append(error_table[serial]["GT"])

    error_table = {
        "Serial": error_table_serial,
        "Pred": error_table_pred,
        "GT": error_table_gt,
    }

    accuracy = confusion_matrix.trace() / confusion_matrix.sum() * 100.0
    return accuracy, score, target, confusion_matrix, error_table, crop_timing


@torch.no_grad()
def check_accuracy_multicrop(model, loader, preprocess, config, aggregation="prob", repeat=1):
    # for accuracy
    correct, total = (0, 0)

    for k in range(repeat):
        for sample_batched in loader:
            # estimate
            if aggregation == "logit":
                s = estimate_logit(model, sample_batched, preprocess, config)
            elif aggregation in ["prob", "probability"]:
                s = estimate_class_score(model, sample_batched, preprocess, config)
            else:
                raise ValueError(
                    f"check_accuracy_multicrop(aggregation): aggregation option must "
                    f"be one among of ['logit', 'prob', 'probability']."
                )
            y = sample_batched["class_label"]
            tcm = config["test_crop_multiple"]

            # multi-crop averaging
            if s.size(0) % tcm != 0:
                raise ValueError(
                    f"check_accuracy_multicrop(): Real minibatch size={y.size(0)} is not multiple of "
                    f"config['test_crop_multiple']={tcm}."
                )

            real_minibatch = s.size(0) // tcm
            s_ = torch.zeros((real_minibatch, s.size(1)))
            y_ = torch.zeros((real_minibatch,), dtype=torch.int32)

            for m in range(real_minibatch):
                s_[m] = s[tcm * m : tcm * (m + 1)].mean(dim=0, keepdims=True)
                y_[m] = y[tcm * m]

            s = s_
            y = y_

            # calculate accuracy
            pred = s.argmax(dim=-1)
            correct += pred.squeeze().eq(y).sum().item()
            total += pred.shape[0]

    accuracy = 100.0 * correct / total
    return accuracy


@torch.no_grad()
def check_accuracy_multicrop_extended(model, loader, preprocess, config, aggregation="prob", repeat=1, dummy=1):
    # for confusion matrix
    C = config["out_dims"]
    confusion_matrix = np.zeros((C, C), dtype=np.int32)

    # for ROC curve
    score = None
    target = None

    # for throughput calculation
    total = 0
    total_time = 0.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # warm-up using dummy round
    for k in range(dummy):
        for sample_batched in loader:
            _ = estimate_class_score(model, sample_batched, preprocess, config)

    for k in range(repeat):
        for sample_batched in loader:
            real_minibatch = sample_batched["signal"].size(0) // config["test_crop_multiple"]
            s_merge = torch.zeros((real_minibatch, config["out_dims"]))
            y_merge = torch.zeros((real_minibatch,), dtype=torch.int32)

            # estimate
            start_event.record()
            if aggregation == "logit":
                s = estimate_logit(model, sample_batched, preprocess, config)
            elif aggregation in ["prob", "probability"]:
                s = estimate_class_score(model, sample_batched, preprocess, config)
            else:
                raise ValueError(
                    f"check_accuracy_multicrop(aggregation): aggregation option must "
                    f"be one among of ['logit', 'prob', 'probability']."
                )
            y = sample_batched["class_label"]
            tcm = config["test_crop_multiple"]

            # multi-crop averaging
            if s.size(0) % tcm != 0:
                raise ValueError(
                    f"check_accuracy_multicrop(): Real minibatch size={y.size(0)} is not multiple of "
                    f"config['test_crop_multiple']={tcm}."
                )

            for m in range(real_minibatch):
                s_merge[m] = s[tcm * m : tcm * (m + 1)].mean(dim=0, keepdims=True)
                y_merge[m] = y[tcm * m]

            end_event.record()
            torch.cuda.synchronize()
            total_time += start_event.elapsed_time(end_event) / 1000

            if aggregation == "logit":
                # s_merge = logit_to_prob(s_merge, config)
                s_merge = F.softmax(s_merge, dim=1)

            s = s_merge
            y = y_merge

            # classification score for drawing ROC curve
            if score is None:
                score = s.detach().cpu().numpy()
                target = y.detach().cpu().numpy()
            else:
                score = np.concatenate((score, s.detach().cpu().numpy()), axis=0)
                target = np.concatenate((target, y.detach().cpu().numpy()), axis=0)

            # confusion matrix
            pred = s.argmax(dim=-1)
            confusion_matrix += calculate_confusion_matrix(pred, y, num_classes=config["out_dims"])

            # total samples
            total += pred.shape[0]

    accuracy = confusion_matrix.trace() / confusion_matrix.sum() * 100.0
    throughput = total / total_time

    return accuracy, score, target, confusion_matrix, throughput
