import os
import json
from copy import deepcopy
from dataclasses import dataclass, asdict

import numpy as np
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


@dataclass
class MultiLabel:
    """Dataclass for EEG multi-property diagnosis label.
    """
    dementia: bool = False
    ad: bool = False
    load: bool = False
    eoad: bool = False
    vd: bool = False
    sivd: bool = False
    ad_vd_mixed: bool = False

    mci: bool = False
    mci_ad: bool = False
    mci_amnestic: bool = False
    mci_amnestic_ef: bool = False
    mci_amnestic_rf: bool = False
    mci_non_amnestic: bool = False
    mci_multi_domain: bool = False
    mci_vascular: bool = False

    normal: bool = False
    cb_normal: bool = False
    smi: bool = False
    hc_normal: bool = False

    ftd: bool = False
    bvftd: bool = False
    language_ftd: bool = False
    semantic_aphasia: bool = False
    non_fluent_aphasia: bool = False

    parkinson_synd: bool = False
    parkinson_disease: bool = False
    parkinson_dementia: bool = False
    other_parkinson_synd: bool = False

    nph: bool = False
    tga: bool = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            assert hasattr(self, k), f'(ERROR) MultiLabel dataclass initialization error - has unknown label: {k}'
            setattr(self, k, v)

    def __repr__(self):
        return self.__class__.__qualname__ + str({k: v for k, v in asdict(self).items() if v is True})

    def get_true_keys(self):
        return [k for k, v in asdict(self).items() if v is True]

    def get_dict(self):
        return asdict(self)

    def get_size(self):
        return len(asdict(self).keys())

    def get_label_types(self):
        return asdict(self).keys()

    def get_label_values(self):
        return asdict(self).values()

    def check(self, label: str):
        return getattr(self, label.lower())

    @staticmethod
    def load_from_string(dx1: str):
        """
        create MultiLabel class instance from string dx1 value.
        :param dx1:
        :return:
        """

        # input sanity check
        assert type(dx1) == str, f'ERROR: load_from_string function input is non-string type: {type(dx1)}'

        label = MultiLabel()

        if dx1 in ['load']:
            label = MultiLabel(dementia=True, ad=True, load=True)
        elif dx1 in ['eoad']:
            label = MultiLabel(dementia=True, ad=True, eoad=True)

        elif dx1 in ['vd', 'vascular dementia', 'sivd']:
            label = MultiLabel(dementia=True, vd=True, sivd=True)
        elif dx1 in ['ad-vd-mixed']:
            label = MultiLabel(dementia=True, ad_vd_mixed=True)

        elif dx1 in ['mci']:
            label = MultiLabel(mci=True)

        elif dx1 in ['ad_mci', 'ad-mci']:
            label = MultiLabel(mci=True, mci_ad=True)
        elif dx1 in ['ad-mci amnestic']:
            label = MultiLabel(mci=True, mci_ad=True, mci_amnestic=True)

        elif dx1 in ['ad-mci (ef)']:
            label = MultiLabel(mci=True, mci_ad=True, mci_amnestic=True, mci_amnestic_ef=True)
        elif dx1 in ['ad-mci (rf)']:
            label = MultiLabel(mci=True, mci_ad=True, mci_amnestic=True, mci_amnestic_rf=True)

        elif dx1 in ['mci amnestic', 'amci']:
            label = MultiLabel(mci=True, mci_amnestic=True)
        elif dx1 in ['mci amnestic multi-domain']:
            label = MultiLabel(mci=True, mci_amnestic=True, mci_multi_domain=True)

        elif dx1 in ['mci_ef', 'mci ef', 'mci(ef)', 'amci (ef)', 'amci(ef)', 'mci encoding failure']:
            label = MultiLabel(mci=True, mci_amnestic=True, mci_amnestic_ef=True)
        elif dx1 in ['mci (ef) multi-domain', 'mci encoding failure multi-domain']:
            label = MultiLabel(mci=True, mci_amnestic=True, mci_amnestic_ef=True, mci_multi_domain=True)

        elif dx1 in ['mci_rf', 'mci rf', 'mci (rf)', 'amci rf', 'mci retrieval failure']:
            label = MultiLabel(mci=True, mci_amnestic=True, mci_amnestic_rf=True)
        elif dx1 in ['mci(rf) multi-domain']:
            label = MultiLabel(mci=True, mci_amnestic=True, mci_amnestic_rf=True, mci_multi_domain=True)

        elif dx1 in ['mci non amnestic', 'mci non-amnestic', 'cind']:
            label = MultiLabel(mci=True, mci_non_amnestic=True)

        elif dx1 in ['vascular mci']:
            label = MultiLabel(mci=True, mci_vascular=True)
        elif dx1 in ['vmci non-amnestic']:
            label = MultiLabel(mci=True, mci_vascular=True, mci_non_amnestic=True)
        elif dx1 in ['vmci(ef)']:
            label = MultiLabel(mci=True, mci_amnestic=True, mci_amnestic_ef=True, mci_vascular=True)
        elif dx1 in ['vmci(rf)', 'vascular mci (rf)']:
            label = MultiLabel(mci=True, mci_amnestic=True, mci_amnestic_rf=True, mci_vascular=True)

        elif dx1 in ['nc', 'nl']:
            label = MultiLabel(normal=True)
        elif dx1 in ['cb_normal']:
            label = MultiLabel(normal=True, cb_normal=True)
        elif dx1 in ['smi']:
            label = MultiLabel(normal=True, smi=True)
        elif dx1 in ['hc_normal']:
            label = MultiLabel(normal=True, hc_normal=True)

        elif dx1 in ['ftd']:
            label = MultiLabel(ftd=True)
        elif dx1 in ['bvftd']:
            label = MultiLabel(ftd=True, bvftd=True)
        elif dx1 in ['language ftd']:
            label = MultiLabel(ftd=True, language_ftd=True)
        elif dx1 in ['semantic aphasia']:
            label = MultiLabel(ftd=True, semantic_aphasia=True)
        elif dx1 in ['non fluent aphasia']:
            label = MultiLabel(ftd=True, non_fluent_aphasia=True)

        elif dx1 in ['parkinson_synd']:
            label = MultiLabel(parkinson_synd=True)
        elif dx1 in ['pd', 'parkinson\'s disease']:
            label = MultiLabel(parkinson_synd=True, parkinson_disease=True)
        elif dx1 in ['pdd', 'parkinson dementia']:
            label = MultiLabel(dementia=True, parkinson_synd=True, parkinson_dementia=True)
        elif dx1 in ['other parkinson synd']:
            label = MultiLabel(parkinson_synd=True, other_parkinson_synd=True)

        elif dx1 in ['nph']:
            label = MultiLabel(nph=True)

        elif dx1 in ['tga']:
            label = MultiLabel(tga=True)

        else:
            if dx1 in ['unknown', '0', '?검사없음']:
                label = MultiLabel()
            else:
                print(f'(Warning) load_from_string function cannot parse the input: {dx1}')

        assert 'label' in dir(), 'load_from_string() - unknown dx1 label: %s' % dx1
        return label


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
    splitted_metadata = [[] for i in diagnosis_filter]

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


class EEGDataset(Dataset):
    """EEG Dataset Class for PyTorch.

    Args:
        root_dir (str): Root path to the EDF data files.
        metadata (list of dict): List of dictionary with metadata.
        transform (callable, optional): Optional transform to be applied on each data.
    """

    def __init__(self, root_dir, metadata, transform=None):
        self.root_dir = root_dir
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        m = self.metadata[idx]
        fname = os.path.join(self.root_dir, m['serial'] + '.npy')
        signal = np.load(fname)
        sample = {'signal': signal,
                  'age': m['age'],
                  'class_label': m['class_label'],
                  'metadata': m}
        if self.transform:
            sample = self.transform(sample)
        return sample


class EEGRandomCrop(object):
    """Randomly crop the EEG data to a given size.

    Args:
        - crop_length (int): Desired output signal length.
        - multiple (int): Desired number of cropping.
    """

    def __init__(self, crop_length: int, multiple: int = 1):
        assert isinstance(crop_length, int), 'EEGRandomCrop.__init__(crop_length) needs a integer to initialize'
        if multiple < 1:
            raise ValueError('EEGRandomCrop.__init__(multiple) needs a positive integer to initialize')

        self.crop_length = crop_length
        self.multiple = multiple

    def _random_crop_signal(self, signal):
        total_length = signal.shape[-1]
        start_point = np.random.randint(total_length - self.crop_length)

        return signal[:, start_point:start_point + self.crop_length]

    def __call__(self, sample):
        signal = sample['signal']

        if self.multiple == 1:
            sample['signal'] = self._random_crop_signal(signal)
        else:
            signals = []
            for r in range(self.multiple):
                signals.append(self._random_crop_signal(signal))
            sample['signal'] = signals

        return sample


class EEGRandomCropDebug(object):
    """Randomly crop the EEG data to a given size (debug version).

    Args:
        - crop_length (int): Desired output signal length.
        - multiple (int): Desired number of cropping.
    """

    def __init__(self, crop_length: int, multiple: int = 1):
        assert isinstance(crop_length, int), 'EEGRandomCropDebug.__init__(crop_length) needs a integer to initialize'
        if multiple < 1:
            raise ValueError('EEGRandomCropDebug.__init__(multiple) needs a positive integer to initialize')

        self.crop_length = crop_length
        self.multiple = multiple

    def _random_crop_signal(self, signal):
        total_length = signal.shape[-1]
        start_point = np.random.randint(total_length - self.crop_length)

        return signal[:, start_point:start_point + self.crop_length], start_point

    def __call__(self, sample):
        signal = sample['signal']

        if self.multiple == 1:
            sample['signal'], sample['metadata']['start_point'] = self._random_crop_signal(signal)
        else:
            signals = []
            start_points = []
            for r in range(self.multiple):
                s, sp = self._random_crop_signal(signal)
                signals.append(s)
                start_points.append(sp)

            sample['signal'] = signals
            sample['metadata']['start_point'] = start_points

        return sample


class EEGEyeOpenCrop(object):
    """Crop the EEG signal around the eye-opened event to a given size.

    Args:
        - crop_before (int): Desired signal length to crop right before the event occurred
        - crop_after (int): Desired signal length to crop right after the event occurred
        - jitter (int): Amount of jitter
        - mode (str, optional): Way for selecting one among multiple same events
    """

    def __init__(self, crop_before: int, crop_after: int, jitter: int, mode: str = 'first'):
        if mode not in ('first', 'random'):
            raise ValueError('EEGEyeOpenCrop.__init__(mode) must be set to one of ("first", "random")')

        self.crop_before = crop_before
        self.crop_after = crop_after
        self.jitter = jitter
        self.mode = mode

    def __call__(self, sample):
        signal = sample['signal']
        total_length = signal.shape[-1]

        candidates = []
        for e in sample['metadata']['events']:
            if e[1].lower() == 'eyes open':
                candidates.append(e[0])

        if len(candidates) == 0:
            raise ValueError(f'EEGEyeOpenCrop.__call__(), {sample["metadata"]["serial"]} does not have an eye open '
                             f'event.')

        if self.mode == 'first':
            candidates = [candidates[0]]

        (neg, pos) = (-self.jitter, self.jitter + 1)

        for k in range(len(candidates)):
            i = np.random.randint(low=0, high=len(candidates))
            time = candidates[i]

            if time - self.crop_before < 0:
                candidates.pop(i)
            elif time - self.crop_before + neg < 0:
                neg = -time + self.crop_before
                break
            elif time + self.crop_after > total_length:
                candidates.pop(i)
            elif time + pos + self.crop_after > total_length:
                pos = total_length - time - self.crop_after
                break
            else:
                break

        if len(candidates) == 0:
            raise ValueError(f'EEGEyeOpenCrop.__call__(), all events of {sample["metadata"]["serial"]} do not '
                             f'have the enough length to crop.')

        time = time + np.random.randint(low=neg, high=pos)
        sample['signal'] = signal[:, time - self.crop_before:time + self.crop_after]
        return sample


class EEGEyeClosedCrop(object):
    """Crop the EEG signal during the eye-closed event to a given size.

    Args:
        - transition (int): Amount of standby until cropping after the eye-closed event occurred
        - crop_length (int): Desired output signal length
        - jitter (int, optional): Amount of jitter
        - mode (str, optional): Way for selecting one during eye-closed
    """

    def __init__(self, transition: int, crop_length: int, jitter: int = 0, mode: str = 'random'):
        if mode not in ('random', 'exact'):
            raise ValueError('EEGEyeClosedCrop.__init__(mode) must be set to one of ("random", "exact")')

        self.transition = transition
        self.crop_length = crop_length
        self.jitter = jitter
        self.mode = mode

    def __call__(self, sample):
        signal = sample['signal']
        total_length = signal.shape[-1]

        intervals = []
        started = True
        opened = False
        t = 0

        for e in sample['metadata']['events']:
            if e[1].lower() == 'eyes open':
                if started:
                    started = False
                    opened = True
                elif opened is False:
                    intervals.append((t, e[0]))
                    opened = True
            elif e[1].lower() == 'eyes closed':
                if started:
                    t = e[0]
                    started = False
                    opened = False
                elif opened:
                    t = e[0]
                    opened = False
                else:
                    t = e[0]

        intervals = [(x[0], min(x[1], total_length))
                     for x in intervals
                     if min(x[1], total_length) - x[0] >= self.transition + self.crop_length + self.jitter]

        if len(intervals) == 0:
            raise ValueError(f'EEGEyeClosedCrop.__call__(), {sample["metadata"]["serial"]} does not have an useful '
                             f'eye-closed event, its total length is {total_length}, and its events are: '
                             f'{sample["metadata"]["events"]}')

        k = np.random.randint(low=0, high=len(intervals))
        if self.mode == 'random':
            t1, t2 = intervals[k]
            if t1 + self.transition == t2 - self.crop_length:
                time = t1 + self.transition
            else:
                time = np.random.randint(low=t1 + self.transition,
                                     high=t2 - self.crop_length)
        elif self.mode == 'exact':
            t1, t2 = intervals[k]
            if self.jitter == 0:
                time = t1 + self.transition
            else:
                time = np.random.randint(low=t1 + self.transition - self.jitter,
                                         high=t1 + self.transition + self.jitter)
        # print(f'{t1:>8d}\t{t2:>8d}\t{time:>8d}\t{time + self.crop_length:>8d}\t{total_length:>8d}')
        sample['signal'] = signal[:, time:time + self.crop_length]
        return sample


class EEGNormalizePerSignal(object):
    """Normalize multichannel EEG signal by its internal statistics."""

    def __init__(self, eps=1e-8):
        self.eps = eps

    def _normalize_per_signal(self, signal):
        if torch.is_tensor(signal):
            std, mean = torch.std_mean(signal, dim=-1, keepdim=True)
        else:
            mean = np.mean(signal, axis=-1, keepdims=True)
            std = np.std(signal, axis=-1, keepdims=True)
        return (signal - mean) / (std + self.eps)

    def __call__(self, sample):
        signal = sample['signal']

        if isinstance(signal, (np.ndarray,)):
            sample['signal'] = self._normalize_per_signal(signal)
        elif isinstance(signal, (list,)):
            signals = []
            for s in signal:
                signals.append(self._normalize_per_signal(s))
            sample['signal'] = signals
        else:
            raise ValueError('EEGNormalizePerSignal.__call__(sample["signal"]) needs to be set to np.ndarray '
                             'or list of np.ndarray')

        return sample


class EEGNormalizeMeanStd(object):
    """Normalize multichannel EEG signal by pre-calculated statistics."""

    def __init__(self, mean, std, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def _normalize_mean_std(self, signal):
        return (signal - self.mean) / (self.std + self.eps)

    def __call__(self, sample):
        signal = sample['signal']

        if isinstance(signal, (np.ndarray,)):
            sample['signal'] = self._normalize_mean_std(signal)
        elif isinstance(signal, (list,)):
            signals = []
            for s in signal:
                signals.append(self._normalize_mean_std(s))
            sample['signal'] = signals
        else:
            raise ValueError('EEGNormalizeMeanStd.__call__(sample["signal"]) needs to be set to np.ndarray '
                             'or list of np.ndarray')
        return sample


class EEGAdditiveGaussianNoise(object):
    """Additive white Gaussian noise."""

    def __init__(self, mean=0.0, std=1e-2):
        self.mean = mean
        self.std = std

    def _add_gaussian_noise(self, signal):
        if torch.is_tensor(signal):
            noise = torch.normal(mean=torch.ones_like(signal)*self.mean,
                                 std=torch.ones_like(signal)*self.std)
        else:
            noise = np.random.normal(loc=self.mean, scale=self.std, size=signal.shape)
        return signal + noise

    def __call__(self, sample):
        signal = sample['signal']

        if isinstance(signal, (np.ndarray,)):
            sample['signal'] = self._add_gaussian_noise(signal)
        elif isinstance(signal, (list,)):
            signals = []
            for s in signal:
                signals.append(self._add_gaussian_noise(s))
            sample['signal'] = signals
        else:
            raise ValueError('EEGAddGaussianNoise.__call__(sample["signal"]) needs to be set to np.ndarray '
                             'or their array')
        return sample


class EEGMultiplicativeGaussianNoise(object):
    """Multiplicative white Gaussian noise."""

    def __init__(self, mean=0.0, std=1e-2):
        self.mean = mean
        self.std = std

    def _add_multiplicative_gaussian_noise(self, signal):
        if torch.is_tensor(signal):
            noise = torch.normal(mean=torch.ones_like(signal)*self.mean,
                                 std=torch.ones_like(signal)*self.std)
        else:
            noise = np.random.normal(loc=self.mean, scale=self.std, size=signal.shape)
        return signal + (signal * noise)

    def __call__(self, sample):
        signal = sample['signal']

        if isinstance(signal, (np.ndarray,)):
            sample['signal'] = self._add_multiplicative_gaussian_noise(signal)
        elif isinstance(signal, (list,)):
            signals = []
            for s in signal:
                signals.append(self._add_multiplicative_gaussian_noise(s))
            sample['signal'] = signals
        else:
            raise ValueError('EEGAddGaussianNoise.__call__(sample["signal"]) needs to be set to np.ndarray '
                             'or their array')
        return sample


class EEGNormalizeAge(object):
    """Normalize age of EEG metadata by the calculated statistics.

    Args:
        - mean: Mean age of all people in EEG training dataset.
        - std: Standard deviation of the age for all people in EEG training dataset.
        - eps: Small number to prevent zero division.
    """

    def __init__(self, mean, std, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, sample):
        sample['age'] = (sample['age'] - self.mean) / (self.std + self.eps)
        return sample


class EEGAddGaussianNoiseAge(object):
    """Add a Gaussian noise to the age value

    Args:
        - mean: Desired mean of noise level for the age value.
        - std: Desired standard deviation of noise level for the age value.
    """

    def __init__(self, mean=0.0, std=1e-2):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        age = sample['age']

        if torch.is_tensor(age):
            noise = torch.normal(mean=torch.ones_like(age)*self.mean,
                                 std=torch.ones_like(age)*self.std)
        else:
            noise = np.random.normal(loc=self.mean, scale=self.std)

        sample['age'] = age + noise
        return sample


class EEGDropEKGChannel(object):
    """Drop the EKG channel from EEG signal."""

    def _drop_ekg_channel(self, signal):
        return np.delete(signal, 19, 0)

    def __call__(self, sample):
        signal = sample['signal']

        if isinstance(signal, (np.ndarray,)):
            sample['signal'] = self._drop_ekg_channel(signal)
        elif isinstance(signal, (list,)):
            signals = []
            for s in signal:
                signals.append(self._drop_ekg_channel(s))
            sample['signal'] = signals
        else:
            raise ValueError('EEGDropEKGChannel.__call__(sample["signal"]) needs to be set to np.ndarray '
                             'or their array')
        return sample


class EEGDropPhoticChannel(object):
    """Drop the photic stimulation channel from EEG signal."""

    def _drop_photic_channel(self, signal):
        return signal[:-1]

    def __call__(self, sample):
        signal = sample['signal']

        if isinstance(signal, (np.ndarray,)):
            sample['signal'] = self._drop_photic_channel(signal)
        elif isinstance(signal, (list,)):
            signals = []
            for s in signal:
                signals.append(self._drop_photic_channel(s))
            sample['signal'] = signals
        else:
            raise ValueError('EEGDropPhoticChannel.__call__(sample["signal"]) needs to be set to np.ndarray '
                             'or their array')
        return sample


class EEGDropSpecificChannel(object):
    """Drop the specified channel from EEG signal."""

    def __init__(self, drop_channel):
        self.drop_channel = drop_channel

    def drop_specific_channel(self, signal):
        return np.delete(signal, self.drop_channel, 0)

    def __call__(self, sample):
        signal = sample['signal']

        if isinstance(signal, (np.ndarray,)):
            sample['signal'] = self.drop_specific_channel(signal)
        elif isinstance(signal, (list,)):
            signals = []
            for s in signal:
                signals.append(self.drop_specific_channel(s))
            sample['signal'] = signals
        else:
            raise ValueError('EEGDropSpecificChannel.__call__(sample["signal"]) needs to be set to np.ndarray '
                             'or their array')
        return sample


class EEGToTensor(object):
    """Convert EEG numpy array in sample to Tensors."""

    def _signal_to_tensor(self, signal):
        return torch.tensor(signal, dtype=torch.float32)

    def __call__(self, sample):
        signal = sample['signal']

        if isinstance(signal, (np.ndarray,)):
            sample['signal'] = self._signal_to_tensor(signal)
        elif isinstance(signal, (list,)):
            signals = []
            for s in signal:
                signals.append(self._signal_to_tensor(s))
            sample['signal'] = signals
        else:
            raise ValueError('EEGToTensor.__call__(sample["signal"]) needs to be set to np.ndarray '
                             'or their array')

        sample['age'] = torch.tensor(sample['age'], dtype=torch.float32)
        sample['class_label'] = torch.tensor(sample['class_label'])

        return sample


class EEGSpectrogram(object):
    """Transform the multi-channel 1D sequence as multi-channel 2D image using short-time fourier transform
    (a.k.a. Spectrogram) """

    def __init__(self, n_fft, complex_mode='as_real', **kwargs):
        if complex_mode not in ('as_real', 'power', 'remove'):
            raise ValueError('complex_mode must be set to one of ("as_real", "power", "remove")')

        self.n_fft = n_fft
        self.complex_mode = complex_mode
        self.stft_kwargs = kwargs

    def _spectrogram(self, signal):
        if torch.is_tensor(signal) is False:
            raise TypeError('Before transforming the data signal as a spectrogram '
                            'it must be converted to a PyTorch Tensor object using EEGToTensor() transform.')

        signal_f = torch.stft(signal, n_fft=self.n_fft, return_complex=True, **self.stft_kwargs)

        if self.complex_mode == 'as_real':
            signal_f1 = torch.view_as_real(signal_f)[..., 0]
            signal_f2 = torch.view_as_real(signal_f)[..., 1]
            signal_f = torch.cat((signal_f1, signal_f2), dim=0)
        elif self.complex_mode == 'complex':
            pass
        elif self.complex_mode == 'power':
            signal_f = signal_f.abs()
        elif self.complex_mode == 'remove':
            signal_f = torch.real(signal_f)

        return signal_f

    def __call__(self, sample):
        signal = sample['signal']

        if torch.is_tensor(signal):
            sample['signal'] = self._spectrogram(signal)
        elif isinstance(signal, (list,)):
            signals = []
            for s in signal:
                signals.append(self._spectrogram(s))
            sample['signal'] = signals
        else:
            raise ValueError('EEGSpectrogram.__call__(sample["signal"]) needs to be set to np.ndarray '
                             'or their array')

        return sample


def eeg_collate_fn(batch):
    signal = []
    age = []
    class_label = []
    metadata = []

    for sample in batch:
        if isinstance(sample['signal'], (np.ndarray,)) or torch.is_tensor(sample['signal']):
            signal += [sample['signal']]
            age += [sample['age']]
            class_label += [sample['class_label']]
            metadata += [sample['metadata']]
        elif isinstance(sample['signal'], (list,)):
            for s in sample['signal']:
                signal += [s]
                age += [sample['age']]
                class_label += [sample['class_label']]
                metadata += [sample['metadata']]

    batched_sample = {'signal': torch.stack(signal),
                      'age': torch.stack(age),
                      'class_label': torch.stack(class_label),
                      'metadata': metadata}
    return batched_sample


def calculate_age_statistics(config, metadata_train, verbose=False):
    ages = np.array([m['age'] for m in metadata_train])
    age_mean = np.mean(ages)
    age_std = np.std(ages)

    if verbose:
        print('Age mean and standard deviation:')
        print(age_mean, age_std)
        print('\n' + '-' * 100 + '\n')

    return age_mean, age_std


def calculate_signal_statistics(config, metadata_train, repeats=5, verbose=False):
    composed = transforms.Compose([EEGRandomCrop(crop_length=config['crop_length'])])
    train_dataset = EEGDataset(config['data_path'], metadata_train, composed)

    signal_means = []
    signal_stds = []

    for i in range(repeats):
        for d in train_dataset:
            signal_means.append(d['signal'].mean(axis=1, keepdims=True))
            signal_stds.append(d['signal'].std(axis=1, keepdims=True))

    signal_mean = np.mean(np.array(signal_means), axis=0)
    signal_std = np.mean(np.array(signal_stds), axis=0)

    if verbose:
        print('Mean and standard deviation for signal:')
        print(signal_mean, '\n\n', signal_std)
        print('\n' + '-' * 100 + '\n')

    return signal_mean, signal_std


def compose_datasets(config, metadata_train, metadata_val, metadata_test, verbose=False):
    composed_train = []
    composed_test = []

    ###############
    # signal crop #
    ###############
    if config.get('evaluation_phase') is True:
        composed_train += [EEGRandomCropDebug(crop_length=config['crop_length'])]
        composed_test += [EEGRandomCropDebug(crop_length=config['crop_length'])]  # can remove the multiple
    else:
        composed_train += [EEGRandomCrop(crop_length=config['crop_length'],
                                         multiple=config.get('crop_multiple', 1))]
        composed_test += [EEGRandomCrop(crop_length=config['crop_length'],
                                        multiple=config.get('crop_multiple', 1))]  # can remove the multiple

    ###############################
    # data normalization (signal) #
    ###############################
    if config['input_norm'] == 'dataset':
        config['signal_mean'], config['signal_std'] = calculate_signal_statistics(config, metadata_train,
                                                                                  repeats=5, verbose=False)
        composed_train += [EEGNormalizeMeanStd(mean=config['signal_mean'],
                                               std=config['signal_std'])]
        composed_test += [EEGNormalizeMeanStd(mean=config['signal_mean'],
                                              std=config['signal_std'])]
    elif config['input_norm'] == 'datapoint':
        composed_train += [EEGNormalizePerSignal()]
        composed_test += [EEGNormalizePerSignal()]
    elif config['input_norm'] == 'no':
        pass
    else:
        raise ValueError(f"config['input_norm'] have to be set to one of ['dataset', 'datapoint', 'no']")

    ############################
    # data normalization (age) #
    ############################
    config['age_mean'], config['age_std'] = calculate_age_statistics(config, metadata_train, verbose=False)
    composed_train += [EEGNormalizeAge(mean=config['age_mean'], std=config['age_std'])]
    composed_test += [EEGNormalizeAge(mean=config['age_mean'], std=config['age_std'])]

    ########################
    # usage of EEG channel #
    ########################
    if config['EKG'] == 'O':
        pass
    elif config['EKG'] == 'X':
        composed_train += [EEGDropEKGChannel()]
        composed_test += [EEGDropEKGChannel()]
    else:
        raise ValueError(f"config['EKG'] have to be set to one of ['O', 'X']")

    ###########################
    # usage of Photic channel #
    ###########################
    if config['photic'] == 'O':
        pass
    elif config['photic'] == 'X':
        composed_train += [EEGDropPhoticChannel()]
        composed_test += [EEGDropPhoticChannel()]
    else:
        raise ValueError(f"config['photic'] have to be set to one of ['O', 'X']")

    #######################################################
    # additive Gaussian noise for augmentation (signal) #
    #######################################################
    if config.get('evaluation_phase') is True:
        pass
    elif config.get('awgn') is None or config['awgn'] <= 1e-12:
        pass
    elif config['awgn'] > 0.0:
        composed_train += [EEGAdditiveGaussianNoise(mean=0.0, std=config['awgn'])]
    else:
        raise ValueError(f"config['awgn'] have to be None or a positive floating point number")

    #####################################################
    # additive Gaussian noise for augmentation (signal) #
    #####################################################
    if config.get('evaluation_phase') is True:
        pass
    elif config.get('mgn') is None or config['mgn'] <= 1e-12:
        pass
    elif config['mgn'] > 0.0:
        composed_train += [EEGMultiplicativeGaussianNoise(mean=0.0, std=config['mgn'])]
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
        composed_train += [EEGAddGaussianNoiseAge(mean=0.0, std=config['awgn_age'])]
    else:
        raise ValueError(f"config['awgn_age'] have to be None or a positive floating point number")

    ###################
    # numpy to tensor #
    ###################
    composed_train += [EEGToTensor()]
    composed_test += [EEGToTensor()]

    #################################################
    # compose new thing for test on longer sequence #
    #################################################
    composed_test_longer = deepcopy(composed_test)
    if config.get('evaluation_phase') is True:
        composed_test_longer[0] = EEGRandomCropDebug(crop_length=config['longer_crop_length'])
    else:
        composed_test_longer[0] = EEGRandomCrop(crop_length=config['longer_crop_length'])

    #####################
    # transform-compose #
    #####################
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

    ################################################
    # wrap the splitted data using PyTorch Dataset #
    ################################################
    train_dataset = EEGDataset(config['data_path'], metadata_train, composed_train)
    val_dataset = EEGDataset(config['data_path'], metadata_val, composed_test)
    test_dataset = EEGDataset(config['data_path'], metadata_test, composed_test)
    test_dataset_longer = EEGDataset(config['data_path'], metadata_test, composed_test_longer)

    if verbose:
        print('train_dataset[0]:')
        print(train_dataset[0])
        print('\n' + '-' * 100 + '\n')

        print('val_dataset[0]:')
        print(val_dataset[0])
        print('\n' + '-' * 100 + '\n')

        print('test_dataset[0]:')
        print(test_dataset[0])
        print('\n' + '-' * 100 + '\n')

        print('test_dataset_longer[0]:')
        print(test_dataset_longer[0])
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
        for i_batch, sample_batched in enumerate(train_loader):
            sample_batched['signal'].to(config['device'])
            sample_batched['age'].to(config['device'])
            sample_batched['class_label'].to(config['device'])

            print(i_batch,
                  sample_batched['signal'].shape,
                  sample_batched['age'].shape,
                  sample_batched['class_label'].shape,
                  len(sample_batched['metadata']))

            if i_batch > 3:
                break
        print('\n' + '-' * 100 + '\n')

    return train_loader, val_loader, test_loader, test_loader_longer


def build_dataset(config, verbose=False):
    with open(config['meta_path'], 'r') as json_file:
        metadata = json.load(json_file)

    diagnosis_filter, class_label_to_type = define_target_task(config, verbose=verbose)

    splitted_metadata = split_metadata(config, metadata, diagnosis_filter, verbose=verbose)

    metadata_train, metadata_val, metadata_test = shuffle_splitted_metadata(config, splitted_metadata,
                                                                            class_label_to_type, verbose=verbose)

    train_dataset, val_dataset, test_dataset, test_dataset_longer = compose_datasets(config,
                                                                                     metadata_train,
                                                                                     metadata_val,
                                                                                     metadata_test,
                                                                                     verbose=verbose)

    train_loader, val_loader, test_loader, test_loader_longer = make_dataloader(config,
                                                                                train_dataset,
                                                                                val_dataset,
                                                                                test_dataset,
                                                                                test_dataset_longer,
                                                                                verbose=verbose)

    return train_loader, val_loader, test_loader, test_loader_longer, class_label_to_type

