from copy import deepcopy
import numpy as np
import random
import json
import pprint
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from .cau_eeg_dataset import CauEegDataset
from .pipeline import EegRandomCrop, EegRandomCropDebug
from .pipeline import EegNormalizeMeanStd, EegNormalizePerSignal
from .pipeline import EegNormalizeAge
from .pipeline import EegDropEKGChannel, EegDropPhoticChannel
from .pipeline import EegAdditiveGaussianNoise, EegMultiplicativeGaussianNoise
from .pipeline import EegAddGaussianNoiseAge
from .pipeline import EegToTensor, EegToDevice
from .pipeline import eeg_collate_fn

# __all__ = []


def define_target_task(config, verbose=False):
    if config['target_task'] == 'Normal, MCI, Dementia':
        # consider only non-vascular symptoms
        if config['vascular'] == 'X':
            diagnosis_filter = [
                # Normal
                {'type': 'Normal',
                 'include': ['normal'],
                 'exclude': []},
                # Non-vascular MCI
                {'type': 'Non-vascular MCI',
                 'include': ['mci'],
                 'exclude': ['mci_vascular']},
                # Non-vascular dementia
                {'type': 'Non-vascular dementia',
                 'include': ['dementia'],
                 'exclude': ['vd']},
            ]
        # consider all cases
        elif config['vascular'] == 'O':
            diagnosis_filter = [
                # Normal
                {'type': 'Normal',
                 'include': ['normal'],
                 'exclude': []},
                # Non-vascular MCI
                {'type': 'MCI',
                 'include': ['mci'],
                 'exclude': []},
                # Non-vascular dementia
                {'type': 'Dementia',
                 'include': ['dementia'],
                 'exclude': []},
            ]
        else:
            raise ValueError(f"config['vascular'] have to be set to one of ['O', 'X']")
    else:
        raise ValueError(f"config['target_task'] have to be set to one of ['Normal, MCI, Dementia']")

    class_label_to_type = [d_f['type'] for d_f in diagnosis_filter]

    if verbose:
        print('class_label_to_type:', class_label_to_type)
        print('\n' + '-' * 100 + '\n')

    return diagnosis_filter, class_label_to_type


def split_metadata(config, metadata, diagnosis_filter, verbose=False):
    def generate_class_label(label):
        for c, f in enumerate(diagnosis_filter):
            inc = set(f['include']) & set(label) == set(f['include'])
            # inc = len(set(f['include']) & set(label)) > 0
            exc = len(set(f['exclude']) & set(label)) == 0
            if inc and exc:
                return c, f['type']
        return -1, 'The others'

    # Split the filtered dataset
    splitted_metadata = [[] for _ in diagnosis_filter]

    for m in metadata:
        c, n = generate_class_label(m['label'])
        if c >= 0:
            m['class_type'] = n
            m['class_label'] = c
            splitted_metadata[c].append(m)

    for i, split in enumerate(splitted_metadata):
        if len(split) == 0:
            raise ValueError(f'(Warning) Split group {i} has no data.')
        elif verbose:
            print(f'- There are {len(split):} data belonging to {split[0]["class_type"]}')

    if verbose:
        print('\n' + '-' * 100 + '\n')

    return splitted_metadata


def shuffle_splitted_metadata(config, splitted_metadata, class_label_to_type, verbose=False):
    # random seed
    random.seed(config['seed'])

    # Train : Val : Test = 8 : 1 : 1
    ratio1 = 0.8
    ratio2 = 0.1

    metadata_train = []
    metadata_val = []
    metadata_test = []

    for split in splitted_metadata:
        random.shuffle(split)

        n1 = round(len(split) * ratio1)
        n2 = n1 + round(len(split) * ratio2)

        metadata_train.extend(split[:n1])
        metadata_val.extend(split[n1:n2])
        metadata_test.extend(split[n2:])

    random.shuffle(metadata_train)
    random.shuffle(metadata_val)
    random.shuffle(metadata_test)

    if verbose:
        train_class_dist = [np.sum([1 for m in metadata_train if m['class_label'] == i])
                            for i in range(len(class_label_to_type))]

        val_class_dist = [np.sum([1 for m in metadata_val if m['class_label'] == i])
                          for i in range(len(class_label_to_type))]

        test_class_dist = [np.sum([1 for m in metadata_test if m['class_label'] == i])
                           for i in range(len(class_label_to_type))]

        print('Train data label distribution\t:', train_class_dist, np.sum(train_class_dist))
        print('Train data label distribution\t:', val_class_dist, np.sum(val_class_dist))
        print('Train data label distribution\t:', test_class_dist, np.sum(test_class_dist))
        print('\n' + '-' * 100 + '\n')

    # restore random seed (stochastic)
    random.seed()

    return metadata_train, metadata_val, metadata_test


def calculate_age_statistics(config, train_loader, verbose=False):
    age_means = []
    age_stds = []

    for sample in train_loader:
        age = sample['age']
        std, mean = torch.std_mean(age, dim=-1, keepdims=True)
        age_means.append(mean)
        age_stds.append(std)

    age_mean = torch.cat(age_means, dim=0).mean(dim=0, keepdims=True)
    age_std = torch.cat(age_stds, dim=0).mean(dim=0, keepdims=True)

    if verbose:
        print('Age mean and standard deviation:')
        print(age_mean, age_std)
        print('\n' + '-' * 100 + '\n')

    return age_mean, age_std


def calculate_signal_statistics(config, train_loader, repeats=5, verbose=False):
    signal_means = []
    signal_stds = []

    for i in range(repeats):
        for sample in train_loader:
            signal = sample['signal']
            std, mean = torch.std_mean(signal, dim=-1, keepdims=True)  # [N, C, 1]
            signal_means.append(mean)
            signal_stds.append(std)

    signal_mean = torch.cat(signal_means, dim=0).mean(dim=0, keepdims=True)  # [B, C, 1] --> [1, C, 1]
    signal_std = torch.cat(signal_stds, dim=0).mean(dim=0, keepdims=True)

    if verbose:
        print('Mean and standard deviation for signal:')
        print(signal_mean, '\n\n', signal_std)
        print('\n' + '-' * 100 + '\n')

    return signal_mean, signal_std


def compose_transforms(config, verbose=False):
    composed_train = []
    composed_test = []

    ########################
    # usage of EEG channel #
    ########################
    if config['EKG'] == 'O':
        pass
    elif config['EKG'] == 'X':
        composed_train += [EegDropEKGChannel()]
        composed_test += [EegDropEKGChannel()]
    else:
        raise ValueError(f"config['EKG'] have to be set to one of ['O', 'X']")

    ###########################
    # usage of Photic channel #
    ###########################
    if config['photic'] == 'O':
        pass
    elif config['photic'] == 'X':
        composed_train += [EegDropPhoticChannel()]
        composed_test += [EegDropPhoticChannel()]
    else:
        raise ValueError(f"config['photic'] have to be set to one of ['O', 'X']")

    ###############
    # signal crop #
    ###############
    if config.get('evaluation_phase') is True:
        composed_train += [EegRandomCropDebug(crop_length=config['crop_length'])]
        composed_test += [EegRandomCropDebug(crop_length=config['crop_length'])]
    else:
        composed_train += [EegRandomCrop(crop_length=config['crop_length'],
                                         multiple=config.get('crop_multiple', 1))]
        composed_test += [EegRandomCrop(crop_length=config['crop_length'],
                                        multiple=config.get('crop_multiple', 1))]  # can add or remove the multiple

    ###################
    # numpy to tensor #
    ###################
    composed_train += [EegToTensor()]
    composed_test += [EegToTensor()]

    #################################################
    # compose new thing for test on longer sequence #
    #################################################
    composed_test_longer = deepcopy(composed_test)
    if config.get('evaluation_phase') is True:
        composed_test_longer[0] = EegRandomCropDebug(crop_length=config['longer_crop_length'])
    else:
        composed_test_longer[0] = EegRandomCrop(crop_length=config['longer_crop_length'])

    #####################
    # transform-compose #
    #####################
    # composed_train = [TransformTimeChecker(t, '', '>50') for t in composed_train]
    # composed_test = [TransformTimeChecker(t, '', '>50') for t in composed_test]
    # composed_test_longer = [TransformTimeChecker(t, '', '>50') for t in composed_test_longer]

    composed_train = transforms.Compose(composed_train)
    composed_test = transforms.Compose(composed_test)
    composed_test_longer = transforms.Compose(composed_test_longer)

    if verbose:
        print('composed_train:', composed_train)
        print('\n' + '-' * 100 + '\n')

        print('composed_test:', composed_test)
        print('\n' + '-' * 100 + '\n')

        print('longer_composed_test:', composed_test_longer)
        print('\n' + '-' * 100 + '\n')
        print()

    return composed_train, composed_test, composed_test_longer


def compose_preprocess(config, train_loader, verbose=True):
    preprocess_train = []
    preprocess_test = []

    #############
    # to device #
    #############
    preprocess_train += [EegToDevice(device=config['device'])]
    preprocess_test += [EegToDevice(device=config['device'])]

    ###############################
    # data normalization (signal) #
    ###############################
    if config['input_norm'] == 'dataset':
        config['signal_mean'], config['signal_std'] = calculate_signal_statistics(config, train_loader,
                                                                                  repeats=5, verbose=False)
        preprocess_train += [EegNormalizeMeanStd(mean=config['signal_mean'],
                                                 std=config['signal_std'])]
        preprocess_test += [EegNormalizeMeanStd(mean=config['signal_mean'],
                                                std=config['signal_std'])]
    elif config['input_norm'] == 'datapoint':
        preprocess_train += [EegNormalizePerSignal()]
        preprocess_test += [EegNormalizePerSignal()]
    elif config['input_norm'] == 'no':
        pass
    else:
        raise ValueError(f"config['input_norm'] have to be set to one of ['dataset', 'datapoint', 'no']")

    ############################
    # data normalization (age) #
    ############################
    config['age_mean'], config['age_std'] = calculate_age_statistics(config, train_loader, verbose=False)
    preprocess_train += [EegNormalizeAge(mean=config['age_mean'], std=config['age_std'])]
    preprocess_test += [EegNormalizeAge(mean=config['age_mean'], std=config['age_std'])]

    #######################################################
    # additive Gaussian noise for augmentation (signal) #
    #######################################################
    if config.get('evaluation_phase') is True:
        pass
    elif config.get('awgn') is None or config['awgn'] <= 1e-12:
        pass
    elif config['awgn'] > 0.0:
        preprocess_train += [EegAdditiveGaussianNoise(mean=0.0, std=config['awgn'])]
    else:
        raise ValueError(f"config['awgn'] have to be None or a positive floating point number")

    ###########################################################
    # multiplicative Gaussian noise for augmentation (signal) #
    ###########################################################
    if config.get('evaluation_phase') is True:
        pass
    elif config.get('mgn') is None or config['mgn'] <= 1e-12:
        pass
    elif config['mgn'] > 0.0:
        preprocess_train += [EegMultiplicativeGaussianNoise(mean=0.0, std=config['mgn'])]
    else:
        raise ValueError(f"config['mgn'] have to be None or a positive floating point number")

    ##################################################
    # additive Gaussian noise for augmentation (age) #
    ##################################################
    if config.get('evaluation_phase') is True:
        pass
    elif config.get('awgn_age') is None or config['awgn_age'] <= 1e-12:
        pass
    elif config['awgn_age'] > 0.0:
        preprocess_train += [EegAddGaussianNoiseAge(mean=0.0, std=config['awgn_age'])]
    else:
        raise ValueError(f"config['awgn_age'] have to be None or a positive floating point number")

    #######################
    # Compose All at Once #
    #######################
    preprocess_train = transforms.Compose(preprocess_train)
    preprocess_train = torch.nn.Sequential(*preprocess_train.transforms)

    preprocess_test = transforms.Compose(preprocess_test)
    preprocess_test = torch.nn.Sequential(*preprocess_test.transforms)

    if verbose:
        print('preprocess_train:', preprocess_train)
        print('\n' + '-' * 100 + '\n')

        print('preprocess_test:', preprocess_test)
        print('\n' + '-' * 100 + '\n')

    return preprocess_train, preprocess_test


def warp_dataset(config, metadata_train, metadata_val, metadata_test,
                 composed_train, composed_test, composed_test_longer, verbose=False):
    #######################################
    # wrap the data using PyTorch Dataset #
    #######################################
    train_dataset = CauEegDataset(config['data_path'],
                                  metadata_train,
                                  config.get('load_event', False),
                                  config.get('file_format', 'memmap'),
                                  composed_train)
    val_dataset = CauEegDataset(config['data_path'],
                                metadata_val,
                                config.get('load_event', False),
                                config.get('file_format', 'memmap'),
                                composed_test)
    test_dataset = CauEegDataset(config['data_path'],
                                 metadata_test,
                                 config.get('load_event', False),
                                 config.get('file_format', 'memmap'),
                                 composed_test)
    test_dataset_longer = CauEegDataset(config['data_path'],
                                        metadata_test,
                                        config.get('load_event', False),
                                        config.get('file_format', 'memmap'),
                                        composed_test_longer)

    if verbose:
        print('train_dataset[0].keys():')
        pprint.pprint(train_dataset[0].keys(), compact=True)
        print('train_dataset[0]["signal"]:')
        pprint.pprint(train_dataset[0]['signal'], compact=True)
        print()
        print('\n' + '-' * 100 + '\n')

        print('val_dataset[0].keys():')
        pprint.pprint(val_dataset[0].keys(), compact=True)
        print('\n' + '-' * 100 + '\n')

        print('test_dataset[0].keys():')
        pprint.pprint(test_dataset[0].keys(), compact=True)
        print('\n' + '-' * 100 + '\n')

        print('test_dataset_longer[0].keys():')
        pprint.pprint(test_dataset_longer[0].keys(), compact=True)
        print('\n' + '-' * 100 + '\n')

    return train_dataset, val_dataset, test_dataset, test_dataset_longer


def make_dataloader(config, train_dataset, val_dataset, test_dataset, test_dataset_longer, verbose=False):
    if config['device'].type == 'cuda':
        num_workers = 0  # A number other than 0 causes an error
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    if config.get('evaluation_phase') is True:
        batch_size = config['minibatch']
    else:
        batch_size = config['minibatch'] // config.get('crop_multiple', 1)

    if config.get('evaluation_phase') is True:
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  collate_fn=eeg_collate_fn)
    else:
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  collate_fn=eeg_collate_fn)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            collate_fn=eeg_collate_fn)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=False,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             collate_fn=eeg_collate_fn)

    test_loader_longer = DataLoader(test_dataset_longer,
                                    batch_size=batch_size // 2,  # to save the memory capacity
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    collate_fn=eeg_collate_fn)

    if verbose:
        print('train_loader:')
        print(train_loader)
        print('\n' + '-' * 100 + '\n')

        print('val_loader:')
        print(val_loader)
        print('\n' + '-' * 100 + '\n')

        print('test_loader:')
        print(test_loader)
        print('\n' + '-' * 100 + '\n')

        print('test_loader_longer:')
        print(test_loader_longer)
        print('\n' + '-' * 100 + '\n')

    return train_loader, val_loader, test_loader, test_loader_longer


def build_dataset(config, verbose=False):
    with open(config['meta_path'], 'r') as json_file:
        metadata = json.load(json_file)

    diagnosis_filter, class_label_to_type = define_target_task(config,
                                                               verbose=verbose)

    splitted_metadata = split_metadata(config,
                                       metadata,
                                       diagnosis_filter,
                                       verbose=verbose)

    metadata_train, metadata_val, metadata_test = shuffle_splitted_metadata(config,
                                                                            splitted_metadata,
                                                                            class_label_to_type,
                                                                            verbose=verbose)

    composed_train, composed_test, composed_test_longer = compose_transforms(config,
                                                                             verbose=verbose)

    train_dataset, val_dataset, test_dataset, test_dataset_longer = warp_dataset(config,
                                                                                 metadata_train,
                                                                                 metadata_val,
                                                                                 metadata_test,
                                                                                 composed_train,
                                                                                 composed_test,
                                                                                 composed_test_longer,
                                                                                 verbose=verbose)

    train_loader, val_loader, test_loader, test_loader_longer = make_dataloader(config,
                                                                                train_dataset,
                                                                                val_dataset,
                                                                                test_dataset,
                                                                                test_dataset_longer,
                                                                                verbose=verbose)

    preprocess_train, preprocess_test = compose_preprocess(config, train_loader, verbose=verbose)

    if verbose:
        for i_batch, sample_batched in enumerate(train_loader):
            # preprocessing includes to-device operation
            preprocess_train(sample_batched)

            print(i_batch,
                  sample_batched['signal'].shape,
                  sample_batched['age'].shape,
                  sample_batched['class_label'].shape,
                  len(sample_batched['metadata']))

            if i_batch > 3:
                break
        print('\n' + '-' * 100 + '\n')

    return [train_loader, val_loader, test_loader, test_loader_longer,
            preprocess_train, preprocess_test, class_label_to_type]
