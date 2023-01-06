import os
import glob
import json
import pprint
import math
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .temple_eeg_dataset import TuhAbnormalDataset
from .caueeg_script import make_dataloader, compose_preprocess
from .pipeline import EegRandomCrop
from .pipeline import EegNormalizeMeanStd, EegNormalizePerSignal
from .pipeline import EegNormalizeAge
from .pipeline import EegDropChannels
from .pipeline import EegAdditiveGaussianNoise, EegMultiplicativeGaussianNoise
from .pipeline import EegAddGaussianNoiseAge
from .pipeline import EegToTensor, EegToDevice
from .pipeline import EegSpectrogram
from .pipeline import eeg_collate_fn


# __all__ = []


def load_tuab_task_datasets(dataset_path: str, file_format: str = 'edf', transform=None, verbose=False):
    """Load the TUAB datasets for the target benchmark task as PyTorch dataset instances.

    Args:
        dataset_path (str): The file path where the dataset files are located.
        task (str): The target task to load among 'dementia' or 'abnormal'.
        file_format (str): Determines which file format will be used (default: 'edf').
        transform (callable): Preprocessing process to apply during loading signals.
        verbose (bool): Whether to print the progress during loading the datasets.

    Returns:
        The PyTorch dataset instances for the train, validation, and test sets for the task and their configurations.
    """
    try:
        extension = file_format
        if file_format == 'memmap':
            extension = 'dat'

        # create validation set
        train_list = []
        val_list = []
        for pathology in ['abnormal', 'normal']:
            for i, file in enumerate(glob.glob(os.path.join(dataset_path, f'train/{pathology}/*.' + extension))):
                if i % 10 == 0:
                    val_list.append(file)
                else:
                    train_list.append(file)
        test_list = glob.glob(os.path.join(dataset_path, f'eval/*/*.' + extension))

        train_dataset = TuhAbnormalDataset(train_list, file_format=file_format, transform=transform)
        val_dataset = TuhAbnormalDataset(val_list, file_format=file_format, transform=transform)
        test_dataset = TuhAbnormalDataset(test_list, file_format=file_format, transform=transform)

    except FileNotFoundError as e:
        print(f"ERROR: load_tuab_task_datasets(dataset_path={dataset_path}) encounters an error of {e}. "
              f"Make sure the dataset path is correct.")
        raise
    except ValueError as e:
        print(f"ERROR: load_tuab_task_datasets(file_format={file_format}) encounters an error of {e}.")
        raise

    config = {
        "task_name": "Temple University Hospital Abnormal Corpus v2.0.0",
        "task_description": "Pathological classification of [Normal] and [Abnormal] EEG",
        "class_label_to_name": [
            "Normal",
            "Abnormal"
        ],
        "class_name_to_label": {
            "Normal": 0,
            "Abnormal": 1
        },
        "signal_header": ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF',
                          'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF',
                          'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF',
                          'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF',
                          'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF',
                          'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF', 'EEG EKG1-REF']
    }

    if verbose:
        print('task config:')
        pprint.pprint(config, compact=True)
        print('\n', '-' * 100, '\n')

        print('train_dataset[0].keys():')
        pprint.pprint(train_dataset[0].keys(), compact=True)

        if torch.is_tensor(train_dataset[0]):
            print('train signal shape:', train_dataset[0]['signal'].shape)
        else:
            print('train signal shape:', train_dataset[0]['signal'][0].shape)
        print('\n' + '-' * 100 + '\n')

        print('val_dataset[0].keys():')
        pprint.pprint(val_dataset[0].keys(), compact=True)
        print('\n' + '-' * 100 + '\n')

        print('test_dataset[0].keys():')
        pprint.pprint(test_dataset[0].keys(), compact=True)
        print('\n' + '-' * 100 + '\n')

    return config, train_dataset, val_dataset, test_dataset


def load_tuab_task_split(dataset_path: str, split: str, file_format: str = 'edf', transform=None, verbose=False):
    """Load the TUAB dataset for the specified split of the target benchmark task as a PyTorch dataset instance.

    Args:
        dataset_path (str): The file path where the dataset files are located.
        split (str): The desired dataset split to get among "train", "validation", and "test".
        load_event (bool): Whether to load the event information occurred during recording EEG signals.
        file_format (str): Determines which file format is used among of EDF, PyArrow Feather, and NumPy memmap.
        transform (callable): Preprocessing process to apply during loading signals.
        verbose (bool): Whether to print the progress during loading the dataset.

    Returns:
        A PyTorch dataset instance for the specified split for the task and their configurations.
    """
    try:
        extension = file_format
        if file_format == 'memmap':
            extension = 'dat'

        if split in ['train', 'val', 'validation']:
            train_list = []
            val_list = []
            for pathology in ['abnormal', 'normal']:
                for i, file in enumerate(glob.glob(os.path.join(dataset_path, f'train/{pathology}/*.' + extension))):
                    if i % 10 == 0:
                        val_list.append(file)
                    else:
                        train_list.append(file)

            if split in 'train':
                dataset = TuhAbnormalDataset(train_list, file_format=file_format, transform=transform)
            else:
                dataset = TuhAbnormalDataset(val_list, file_format=file_format, transform=transform)
        elif split in ['test', 'eval', 'evaluation']:
            test_list = glob.glob(os.path.join(dataset_path, f'eval/*/*.' + extension))
            dataset = TuhAbnormalDataset(test_list, file_format=file_format, transform=transform)
        else:
            raise ValueError(f'split is unknown: {split}.')

    except FileNotFoundError as e:
        print(f"ERROR: load_tuab_task_split(dataset_path={dataset_path}) encounters an error of {e}. "
              f"Make sure the dataset path is correct.")
        raise
    except ValueError as e:
        print(f"ERROR: load_tuab_task_split(file_format={file_format}) encounters an error of {e}.")
        raise

    config = {
        "task_name": "Temple University Hospital Abnormal Corpus v2.0.0",
        "task_description": "Pathological classification of [Normal] and [Abnormal] EEG",
        "class_label_to_name": [
            "Normal",
            "Abnormal"
        ],
        "class_name_to_label": {
            "Normal": 0,
            "Abnormal": 1
        },
        "signal_header": ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF',
                          'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF',
                          'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF',
                          'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF',
                          'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF',
                          'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF', 'EEG EKG1-REF']
    }

    if verbose:
        print('task config:')
        pprint.pprint(config, compact=True)
        print('\n', '-' * 100, '\n')

        print('dataset[0].keys():')
        pprint.pprint(dataset[0].keys(), compact=True)

        if torch.is_tensor(dataset[0]):
            print('signal shape:', dataset[0]['signal'].shape)
        else:
            print('signal shape:', dataset[0]['signal'][0].shape)
        print('\n' + '-' * 100 + '\n')

    return config, dataset


def compose_tuab_transforms(config, verbose=False):
    transform = []
    transform_multicrop = []

    ###############
    # signal crop #
    ###############
    transform += [EegRandomCrop(crop_length=config['seq_length'],
                                length_limit=config.get('signal_length_limit', 10 ** 7),
                                multiple=config.get('crop_multiple', 1),
                                latency=config.get('latency', 0),
                                segment_simulation=config.get('segment_simulation', False),
                                return_timing=config.get('crop_timing_analysis', False),
                                reject_events=config.get('reject_events', False))]
    transform_multicrop += [EegRandomCrop(crop_length=config['seq_length'],
                                          length_limit=config.get('signal_length_limit', 10 ** 7),
                                          multiple=config.get('test_crop_multiple', 8),
                                          latency=config.get('latency', 0),
                                          segment_simulation=config.get('segment_simulation', False),
                                          return_timing=config.get('crop_timing_analysis', False),
                                          reject_events=config.get('reject_events', False))]

    ###################################
    # usage of EKG or photic channels #
    ###################################
    channel_ekg = config['signal_header'].index('EEG EKG1-REF')

    if config['EKG'] == 'O':
        pass
    elif config['EKG'] == 'X':
        transform += [EegDropChannels([channel_ekg])]
        transform_multicrop += [EegDropChannels([channel_ekg])]
    else:
        raise ValueError(f"Both config['EKG'] have to be set to one of ['O', 'X']")

    ###################
    # numpy to tensor #
    ###################
    transform += [EegToTensor()]
    transform_multicrop += [EegToTensor()]

    #####################
    # transform-compose #
    #####################
    # transform = [TransformTimeChecker(t, '', '>50') for t in transform]
    # transform_multicrop = [TransformTimeChecker(t, '', '>50') for t in transform_multicrop]

    transform = transforms.Compose(transform)
    transform_multicrop = transforms.Compose(transform_multicrop)

    if verbose:
        print('transform:', transform)
        print('\n' + '-' * 100 + '\n')

        print('transform_multicrop:', transform_multicrop)
        print('\n' + '-' * 100 + '\n')
        print()

    return transform, transform_multicrop


def build_dataset_for_tuab_train(config, verbose=False):
    dataset_path = config['dataset_path']
    if 'cwd' in config:
        dataset_path = os.path.join(config['cwd'], dataset_path)

    if 'run_mode' not in config.keys():
        print('\n' + '=' * 80 + '\n' + '=' * 80 + '\n')
        print('WARNING: run_mode is not specified.\n \t==> run_mode is set to "train" automatically.')
        print('\n' + '=' * 80 + '\n' + '=' * 80 + '\n')
        config['run_mode'] = 'train'

    config_task, _ = load_tuab_task_split(dataset_path=dataset_path,
                                          split='test',
                                          file_format=config['file_format'])
    config.update(**config_task)

    transform, transform_multicrop = compose_tuab_transforms(config, verbose=verbose)
    config['transform'] = transform
    config['transform_multicrop'] = transform_multicrop

    config_task, train_dataset, val_dataset, test_dataset = load_tuab_task_datasets(dataset_path=dataset_path,
                                                                                    file_format=config['file_format'],
                                                                                    transform=transform,
                                                                                    verbose=verbose)

    config.update(**config_task)

    _, multicrop_test_dataset = load_tuab_task_split(dataset_path=dataset_path,
                                                     split='test',
                                                     file_format=config['file_format'],
                                                     transform=transform_multicrop,
                                                     verbose=verbose)

    train_loader, val_loader, test_loader, multicrop_test_loader = make_dataloader(config,
                                                                                   train_dataset,
                                                                                   val_dataset,
                                                                                   test_dataset,
                                                                                   multicrop_test_dataset,
                                                                                   verbose=False)

    preprocess_train, preprocess_test = compose_preprocess(config, train_loader, verbose=verbose)
    config['preprocess_train'] = preprocess_train
    config['preprocess_test'] = preprocess_test
    config['in_channels'] = preprocess_train(next(iter(train_loader)))['signal'].shape[1]
    config['out_dims'] = len(config['class_label_to_name'])

    if verbose:
        for i_batch, sample_batched in enumerate(train_loader):
            # preprocessing includes to-device operation
            preprocess_train(sample_batched)

            print(i_batch,
                  sample_batched['signal'].shape,
                  sample_batched['age'].shape,
                  sample_batched['class_label'].shape)

            if i_batch > 3:
                break
        print('\n' + '-' * 100 + '\n')

    return train_loader, val_loader, test_loader, multicrop_test_loader
