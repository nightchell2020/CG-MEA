import numpy as np
import torch


class EegRandomCrop(object):
    """Randomly crop the EEG data to a given size.

    Args:
        - crop_length (int): Desired output signal length.
        - multiple (int): Desired number of cropping.
    """

    def __init__(self, crop_length: int, multiple: int = 1):
        assert isinstance(crop_length, int), 'EegRandomCrop.__init__(crop_length) needs a integer to initialize'
        if multiple < 1:
            raise ValueError('EegRandomCrop.__init__(multiple) needs a positive integer to initialize')

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


class EegRandomCropDebug(object):
    """Randomly crop the EEG data to a given size (debug version).

    Args:
        - crop_length (int): Desired output signal length.
        - multiple (int): Desired number of cropping.
    """

    def __init__(self, crop_length: int, multiple: int = 1):
        assert isinstance(crop_length, int), 'EegRandomCropDebug.__init__(crop_length) needs a integer to initialize'
        if multiple < 1:
            raise ValueError('EegRandomCropDebug.__init__(multiple) needs a positive integer to initialize')

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


class EegEyeOpenCrop(object):
    """Crop the EEG signal around the eye-opened event to a given size.

    Args:
        - crop_before (int): Desired signal length to crop right before the event occurred
        - crop_after (int): Desired signal length to crop right after the event occurred
        - jitter (int): Amount of jitter
        - mode (str, optional): Way for selecting one among multiple same events
    """

    def __init__(self, crop_before: int, crop_after: int, jitter: int, mode: str = 'first'):
        if mode not in ('first', 'random'):
            raise ValueError('EegEyeOpenCrop.__init__(mode) must be set to one of ("first", "random")')

        self.crop_before = crop_before
        self.crop_after = crop_after
        self.jitter = jitter
        self.mode = mode

    def __call__(self, sample):
        if 'event' not in sample['metadata'].keys():
            raise ValueError(f'EegEyeOpenCrop.__call__(), this dataset does not have the event information at all.')
        
        signal = sample['signal']
        total_length = signal.shape[-1]

        candidates = []
        for e in sample['metadata']['event']:
            if e[1].lower() == 'eyes open':
                candidates.append(e[0])

        if len(candidates) == 0:
            raise ValueError(f'EegEyeOpenCrop.__call__(), {sample["metadata"]["serial"]} does not have an eye open '
                             f'event.')

        if self.mode == 'first':
            candidates = [candidates[0]]

        (neg, pos) = (-self.jitter, self.jitter + 1)
        time = 0

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
            raise ValueError(f'EegEyeOpenCrop.__call__(), all events of {sample["metadata"]["serial"]} do not '
                             f'have the enough length to crop.')

        time = time + np.random.randint(low=neg, high=pos)
        sample['signal'] = signal[:, time - self.crop_before:time + self.crop_after]
        return sample


class EegEyeClosedCrop(object):
    """Crop the EEG signal during the eye-closed event to a given size.

    Args:
        - transition (int): Amount of standby until cropping after the eye-closed event occurred
        - crop_length (int): Desired output signal length
        - jitter (int, optional): Amount of jitter
        - mode (str, optional): Way for selecting one during eye-closed
    """

    def __init__(self, transition: int, crop_length: int, jitter: int = 0, mode: str = 'random'):
        if mode not in ('random', 'exact'):
            raise ValueError('EegEyeClosedCrop.__init__(mode) must be set to one of ("random", "exact")')

        self.transition = transition
        self.crop_length = crop_length
        self.jitter = jitter
        self.mode = mode

    def __call__(self, sample):
        if 'event' not in sample['metadata'].keys():
            raise ValueError(f'EegEyeClosedCrop.__call__(), this dataset does not have the event information at all.')

        signal = sample['signal']
        total_length = signal.shape[-1]

        intervals = []
        started = True
        opened = False
        t = 00

        for e in sample['metadata']['event']:
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
            raise ValueError(f'EegGEyeClosedCrop.__call__(), {sample["metadata"]["serial"]} does not have an useful '
                             f'eye-closed event, its total length is {total_length}, and its events are: '
                             f'{sample["metadata"]["event"]}')

        k = np.random.randint(low=0, high=len(intervals))
        time = 0

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


class EegNormalizePerSignal(object):
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
            raise ValueError('EegNormalizePerSignal.__call__(sample["signal"]) needs to be set to np.ndarray '
                             'or list of np.ndarray')

        return sample


class EegNormalizeMeanStd(object):
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
            raise ValueError('EegNormalizeMeanStd.__call__(sample["signal"]) needs to be set to np.ndarray '
                             'or list of np.ndarray')
        return sample


class EegAdditiveGaussianNoise(object):
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
            raise ValueError('EegGAddGaussianNoise.__call__(sample["signal"]) needs to be set to np.ndarray '
                             'or their array')
        return sample


class EegMultiplicativeGaussianNoise(object):
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
            raise ValueError('EegAddGaussianNoise.__call__(sample["signal"]) needs to be set to np.ndarray '
                             'or their array')
        return sample


class EegNormalizeAge(object):
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


class EegAddGaussianNoiseAge(object):
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


class EegDropEKGChannel(object):
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


class EegDropPhoticChannel(object):
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
            raise ValueError('EegDropPhoticChannel.__call__(sample["signal"]) needs to be set to np.ndarray '
                             'or their array')
        return sample


class EegDropSpecificChannel(object):
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
            raise ValueError('EegDropSpecificChannel.__call__(sample["signal"]) needs to be set to np.ndarray '
                             'or their array')
        return sample


class EegToTensor(object):
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
            raise ValueError('EegToTensor.__call__(sample["signal"]) needs to be set to np.ndarray '
                             'or their array')

        sample['age'] = torch.tensor(sample['age'], dtype=torch.float32)
        sample['class_label'] = torch.tensor(sample['class_label'])

        return sample


class EegSpectrogram(object):
    """Transform the multichannel 1D sequence as multichannel 2D image using short-time fourier transform
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
                            'it must be converted to a PyTorch Tensor object using EegToTensor() transform.')

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
            raise ValueError('EegSpectrogram.__call__(sample["signal"]) needs to be set to np.ndarray '
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
