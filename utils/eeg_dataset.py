import os
import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass, asdict


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
        elif dx1 in ['mci encoding failure multi-domain']:
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
        elif dx1 in ['vmci(rf)']:
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
    """

    def __init__(self, crop_length):
        assert isinstance(crop_length, int), 'EEGRandomCrop.__init__() needs a integer to initialize'
        self.crop_length = crop_length

    def __call__(self, sample):
        signal = sample['signal']

        total_length = signal.shape[1]
        start_point = np.random.randint(total_length - self.crop_length)

        sample['signal'] = signal[:, start_point:start_point + self.crop_length]
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
        total_length = signal.shape[1]

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
        total_length = signal.shape[1]

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
    """Normalize multi-channel EEG signal by its internal statistics."""

    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, sample):
        signal = sample['signal']

        if torch.is_tensor(signal):
            std, mean = torch.std_mean(signal, dim=1, keepdim=True)
        else:
            mean = np.mean(signal, axis=1, keepdims=True)
            std = np.std(signal, axis=1, keepdims=True)

        sample['signal'] = (signal - mean) / (std + self.eps)
        return sample


class EEGNormalizeMeanStd(object):
    """Normalize multi-channel EEG signal by pre-calculated statistics."""

    def __init__(self, mean, std, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, sample):
        signal = sample['signal']

        sample['signal'] = (signal - self.mean) / (self.std + self.eps)
        return sample


class EEGAddGaussianNoise(object):
    """Additive white Gaussian noise."""

    def __init__(self, mean=0.0, std=1e-2):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        signal = sample['signal']

        if torch.is_tensor(signal):
            noise = torch.normal(mean=torch.ones_like(signal)*self.mean,
                                 std=torch.ones_like(signal)*self.std)
        else:
            noise = np.random.normal(loc=self.mean, scale=self.std, size=signal.shape)

        sample['signal'] = signal + noise
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


class EEGDropEKGChannel(object):
    """Drop the EKG channel from EEG signal."""

    def __call__(self, sample):
        sample['signal'] = np.delete(sample['signal'], 19, 0)
        return sample


class EEGDropPhoticChannel(object):
    """Drop the photic stimulation channel from EEG signal."""

    def __call__(self, sample):
        sample['signal'] = sample['signal'][:-1]
        return sample


class EEGDropSpecificChannel(object):
    """Drop the specified channel from EEG signal."""
    def __init__(self, drop_channel):
        self.drop_channel = drop_channel

    def __call__(self, sample):
        sample['signal'] = np.delete(sample['signal'], self.drop_channel, 0)
        return sample


class EEGToTensor(object):
    """Convert EEG numpy array in sample to Tensors."""

    def __call__(self, sample):
        sample['signal'] = torch.tensor(sample['signal'], dtype=torch.float32)
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

    def __call__(self, sample):
        signal = sample['signal']

        if torch.is_tensor(signal) is False:
            raise TypeError('Before transforming the data signal as a spectrogram '
                            'it must be converted to a PyTorch Tensor object using EEGToTensor() transform.')

        signal = sample['signal']
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

        sample['signal'] = signal_f
        return sample


def eeg_collate_fn(batch):
    signal = []
    age = []
    class_label = []
    metadata = []

    for sample in batch:
        signal += [sample['signal']]
        age += [sample['age']]
        class_label += [sample['class_label']]
        metadata += [sample['metadata']]

    batched_sample = {'signal': torch.stack(signal).contiguous(),
                      'age': torch.stack(age).contiguous(),
                      'class_label': torch.stack(class_label).contiguous(),
                      'metadata': metadata}
    return batched_sample


