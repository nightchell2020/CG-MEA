from typing import Optional
import time
from packaging import version

import numpy as np
import torch
import torchaudio


def make_variable_repr(a: dict):
    return f"({', '.join([f'{k}={v!r}' for k, v in a.items() if not (k.startswith('_') or k == 'training')])})"


class EegLimitMaxLength(object):
    """Cut off the start and end signals by the specified amount.

    Args:
        max_length (int): Signal length limit to cut out the rest.
    """

    def __init__(self, max_length: int):
        if isinstance(max_length, int) is False or max_length <= 0:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(front_cut) " f"needs a positive integer to initialize"
            )
        self.max_length = max_length

    def __call__(self, sample):
        sample["signal"] = sample["signal"][:, : self.max_length]
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


class EegRandomCrop(object):
    """Randomly crop the EEG data to a given size.

    Args:
        crop_length (int): Desired output signal length.
        length_limit (int, optional): Signal length limit to use.
        multiple (int, optional): Desired number of cropping.
        latency (int, optional): Latency signal length to exclude after record starting.
        return_timing (bool, optional): Decide whether to return the sample timing.
        reject_events (bool, optional): Decide whether to reject the segments with events.
    """

    def __init__(
        self,
        crop_length: int,
        length_limit: int = 10**7,
        multiple: int = 1,
        latency: int = 0,
        segment_simulation=False,
        return_timing: bool = False,
        reject_events: bool = False,
    ):
        if isinstance(crop_length, int) is False:
            raise ValueError(f"{self.__class__.__name__}.__init__(crop_length) " f"needs a integer to initialize")
        if isinstance(crop_length, int) is False or multiple < 1:
            raise ValueError(f"{self.__class__.__name__}.__init__(multiple)" f" needs a positive integer to initialize")
        if isinstance(latency, int) is False or latency < 0:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(latency)" f" needs a non negative integer to initialize"
            )

        self.crop_length = crop_length
        self.length_limit = length_limit
        self.multiple = multiple
        self.latency = latency
        self.segment_simulation = segment_simulation
        self.return_timing = return_timing
        self.reject_events = reject_events

    def __call__(self, sample):
        if self.reject_events and "event" not in sample.keys():
            raise ValueError(f"{self.__class__.__name__}, this dataset " f"does not have the event information at all.")

        signal = sample["signal"]
        signal_length = min(signal.shape[-1], self.length_limit)

        possible_timeline = np.ones((signal_length,), dtype=np.int32)
        possible_timeline[: self.latency] = 0
        possible_timeline[-self.crop_length + 1 :] = 0
        if self.reject_events:
            for e in sample["event"]:
                start = max(e[0] - self.crop_length + 1, 0)
                possible_timeline[start : e[0] + 1] = 0

        cts = np.random.choice(np.arange(signal_length)[possible_timeline == 1], self.multiple)

        if self.multiple == 1:
            ct = cts[0]
            if self.segment_simulation:
                ct = int((ct - self.latency) / self.crop_length) * self.crop_length + self.latency

            sample["signal"] = signal[:, ct : ct + self.crop_length]

            if self.return_timing:
                sample["crop_timing"] = ct

            if "event" in sample.keys():
                event = []
                for e in sample["event"]:
                    if ct <= e[0] < ct + self.crop_length:
                        event.append((e[0] - ct, e[1]))
                sample["event"] = event

        else:
            signals = []
            crop_timings = []
            events = []

            for r in range(self.multiple):
                ct = cts[r]
                if self.segment_simulation:
                    ct = int((ct - self.latency) / self.crop_length) * self.crop_length + self.latency

                signals.append(signal[:, ct : ct + self.crop_length])

                if self.return_timing:
                    crop_timings.append(ct)

                if "event" in sample.keys():
                    event = []
                    for e in sample["event"]:
                        if ct <= e[0] < ct + self.crop_length:
                            event.append((e[0] - ct, e[1]))
                    events.append(event)

            sample["signal"] = signals
            if self.return_timing:
                sample["crop_timing"] = crop_timings
            if "event" in sample.keys():
                sample["event"] = events

        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


class EegEyeOpenCrop(object):
    """Crop the EEG signal around the eye-opened event to a given size.

    Args:
        crop_before (int): Desired signal length to crop right before the event occurred
        crop_after (int): Desired signal length to crop right after the event occurred
        jitter (int): Amount of jitter
        mode (str, optional): Way for selecting one among multiple same events
        length_limit (int, optional): Signal length limit to use.
    """

    def __init__(
        self,
        crop_before: int,
        crop_after: int,
        jitter: int,
        mode: str = "first",
        length_limit: int = 10**7,
    ):
        if mode not in ("first", "random"):
            raise ValueError(f"{self.__class__.__name__}.__init__(mode) " f'must be set to one of ("first", "random")')

        self.crop_before = crop_before
        self.crop_after = crop_after
        self.jitter = jitter
        self.mode = mode
        self.length_limit = length_limit

    def __call__(self, sample):
        if "event" not in sample.keys():
            raise ValueError(f"{self.__class__.__name__}, this dataset " f"does not have the event information at all.")

        signal = sample["signal"]
        total_length = min(signal.shape[-1], self.length_limit)

        candidates = []
        for e in sample["event"]:
            if e[1].lower() == "eyes open":
                candidates.append(e[0])

        if len(candidates) == 0:
            raise ValueError(
                f'{self.__class__.__name__}.__call__(), {sample["serial"]} ' f"does not have an eye open event."
            )

        if self.mode == "first":
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
            raise ValueError(
                f'{self.__class__.__name__}.__call__(), all events of {sample["serial"]} '
                f"do not have the enough length to crop."
            )

        time = time + np.random.randint(low=neg, high=pos)
        sample["signal"] = signal[:, time - self.crop_before : time + self.crop_after]
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


class EegEyeClosedCrop(object):
    """Crop the EEG signal during the eye-closed event to a given size.

    Args:
        transition (int): Amount of standby until cropping after the eye-closed event occurred
        crop_length (int): Desired output signal length
        jitter (int, optional): Amount of jitter
        mode (str, optional): Way for selecting one during eye-closed
        length_limit (int, optional): Signal length limit to use.
    """

    def __init__(
        self,
        transition: int,
        crop_length: int,
        jitter: int = 0,
        mode: str = "random",
        length_limit: int = 10 * 7,
    ):
        if mode not in ("random", "exact"):
            raise ValueError(f"{self.__class__.__name__}.__init__(mode) " f'must be set to one of ("random", "exact")')

        self.transition = transition
        self.crop_length = crop_length
        self.jitter = jitter
        self.mode = mode
        self.length_limit = length_limit

    def __call__(self, sample):
        if "event" not in sample.keys():
            raise ValueError(
                f"{self.__class__.__name__}.__call__(), this dataset does not have " f"an event information at all."
            )

        signal = sample["signal"]
        total_length = min(signal.shape[-1], self.length_limit)

        intervals = []
        started = True
        opened = False
        t = 00

        for e in sample["event"]:
            if e[1].lower() == "eyes open":
                if started:
                    started = False
                    opened = True
                elif opened is False:
                    intervals.append((t, e[0]))
                    opened = True
            elif e[1].lower() == "eyes closed":
                if started:
                    t = e[0]
                    started = False
                    opened = False
                elif opened:
                    t = e[0]
                    opened = False
                else:
                    t = e[0]

        intervals = [
            (x[0], min(x[1], total_length))
            for x in intervals
            if min(x[1], total_length) - x[0] >= self.transition + self.crop_length + self.jitter
        ]

        if len(intervals) == 0:
            raise ValueError(
                f'{self.__class__.__name__}.__call__(), {sample["serial"]} does not have '
                f"an useful eye-closed event, its total length is {total_length}, and its events are: "
                f'{sample["event"]}'
            )

        k = np.random.randint(low=0, high=len(intervals))
        time = 0

        if self.mode == "random":
            t1, t2 = intervals[k]
            if t1 + self.transition == t2 - self.crop_length:
                time = t1 + self.transition
            else:
                time = np.random.randint(low=t1 + self.transition, high=t2 - self.crop_length)
        elif self.mode == "exact":
            t1, t2 = intervals[k]
            if self.jitter == 0:
                time = t1 + self.transition
            else:
                time = np.random.randint(
                    low=t1 + self.transition - self.jitter,
                    high=t1 + self.transition + self.jitter,
                )
        # print(f'{t1:>8d}\t{t2:>8d}\t{time:>8d}\t{time + self.crop_length:>8d}\t{total_length:>8d}')
        sample["signal"] = signal[:, time : time + self.crop_length]
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


class EegChangeMontageOrder(object):
    """Transpose the channel order of EEG signal.

    Args:
        ref_montage (list or ndarray): Channel montage of reference
        new_montage (list or ndarray): Channel montage of target
    """

    def __init__(self, ref_montage, new_montage):
        self._calculate_channel_change(ref_montage, new_montage)

    def _calculate_channel_change(self, ref_montage, new_montage):
        self.channel_change = -np.ones((len(new_montage),), dtype="int32")

        for r, ref in enumerate([shl.split("-")[0].lower() for shl in ref_montage]):
            for n, new in enumerate([shl.split("-")[0].lower() for shl in new_montage]):
                other_conditions = (
                    (ref == "t3" and new == "t7")
                    or (ref == "t4" and new == "t8")
                    or (ref == "t5" and new == "p7")
                    or (ref == "t6" and new == "p8")
                    or (ref == "ekg" and new == "ekg1")
                )
                if ref == new or other_conditions:
                    self.channel_change[n] = r

    def _change_channels(self, signal):
        return signal[self.channel_change]

    def __call__(self, sample):
        signal = sample["signal"]

        if isinstance(signal, (list,)):
            signals = []
            for s in signal:
                signals.append(self._change_channels(s))
            sample["signal"] = signals
        else:
            sample["signal"] = self._change_channels(signal)

        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


class EegDropChannels(object):
    """Drop the specified channel from EEG signal.

    Args:
        index (int or list): Channel index(or induce) to drop.
    """

    def __init__(self, index):
        self.drop_index = index

    def drop_specific_channel(self, signal):
        return np.delete(signal, self.drop_index, axis=0)

    def __call__(self, sample):
        signal = sample["signal"]

        if isinstance(signal, (list,)):
            signals = []
            for s in signal:
                signals.append(self.drop_specific_channel(s))
            sample["signal"] = signals
        else:
            sample["signal"] = self.drop_specific_channel(signal)

        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


class EegChannelDifference(object):
    """Drop the specified channel from EEG signal.

    Args:
        ch1 (int): Channel index of interest
        ch2 (int): Channel index of interest
    """

    def __init__(self, ch1, ch2):
        self.ch1 = ch1
        self.ch2 = ch2

    def __call__(self, sample):
        signal = sample["signal"]

        if isinstance(signal, (list,)):
            signals = []
            for s in signal:
                signals.append(s[[self.ch1]] - s[[self.ch2]])
            sample["signal"] = signals
        else:
            sample["signal"] = signal[[self.ch1]] - signal[[self.ch2]]

        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


class EegToTensor(object):
    """Convert EEG numpy array in sample to Tensors."""

    @staticmethod
    def _signal_to_tensor(signal):
        if isinstance(signal, (np.core.memmap,)):
            return torch.tensor(signal).to(dtype=torch.float32)
        return torch.from_numpy(signal).to(dtype=torch.float32)

    def __call__(self, sample):
        signal = sample["signal"]

        if isinstance(signal, (np.ndarray,)):
            sample["signal"] = self._signal_to_tensor(signal)
        elif isinstance(signal, (list,)):
            signals = []
            for s in signal:
                signals.append(self._signal_to_tensor(s))
            sample["signal"] = signals
        else:
            raise ValueError(
                f'{self.__class__.__name__}.__call__(sample["signal"]) needs to be set to np.ndarray ' f"or their list"
            )

        sample["age"] = torch.tensor(sample["age"], dtype=torch.float32)
        if "class_label" in sample.keys():
            sample["class_label"] = torch.tensor(sample["class_label"])

        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


def eeg_collate_fn(batch):
    batched_sample = {k: [] for k in batch[0].keys()}

    for sample in batch:
        if isinstance(sample["signal"], (np.ndarray,)) or torch.is_tensor(sample["signal"]):
            for k in sample.keys():
                batched_sample[k] += [sample[k]]

        elif isinstance(sample["signal"], (list,)):
            multiple = len(sample["signal"])

            for s in sample["signal"]:
                batched_sample["signal"] += [s]

            for k in sample.keys():
                if k not in ["signal", "crop_timing"]:
                    batched_sample[k] += multiple * [sample[k]]
                elif k == "crop_timing":
                    batched_sample[k] += [*sample[k]]

    batched_sample["signal"] = torch.stack(batched_sample["signal"])
    batched_sample["age"] = torch.stack(batched_sample["age"])
    if "class_label" in batched_sample.keys():
        batched_sample["class_label"] = torch.stack(batched_sample["class_label"])

    return batched_sample


class EegToDevice(torch.nn.Module):
    """Add a Gaussian noise to the age value

    Args:
        device: Desired working device.
    """

    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, sample):
        sample["signal"] = sample["signal"].to(self.device)
        sample["age"] = sample["age"].to(self.device)
        if "class_label" in sample.keys():
            sample["class_label"] = sample["class_label"].to(self.device)
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


class EegAverageMontage(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, sample):
        signal = sample["signal"]
        signal[:, self.channels, :] -= torch.mean(signal[:, self.channels, :], dim=-2, keepdim=True)
        sample["signal"] = signal
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


class EegNormalizePerSignal(torch.nn.Module):
    """Normalize multichannel EEG signal by its internal statistics."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, sample):
        signal = sample["signal"]
        std, mean = torch.std_mean(signal, dim=-1, keepdim=True)
        signal.sub_(mean).div_(std + self.eps)
        sample["signal"] = signal
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


class EegNormalizeMeanStd(torch.nn.Module):
    """Normalize multichannel EEG signal by pre-calculated statistics."""

    def __init__(self, mean, std, eps=1e-8):
        super().__init__()

        if isinstance(mean, np.ndarray):
            self.mean = torch.from_numpy(mean)
        elif isinstance(mean, list):
            self.mean = torch.tensor(mean)
        elif torch.is_tensor(mean):
            self.mean = mean
        else:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(mean) needs to be set to among of torch.tensor, "
                f"np.ndarray, or list"
            )

        if isinstance(std, np.ndarray):
            self.std = torch.from_numpy(std)
        elif isinstance(std, list):
            self.std = torch.tensor(std)
        elif torch.is_tensor(std):
            self.std = std
        else:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(std) needs to be set to among of torch.tensor, "
                f"np.ndarray, or list"
            )
        self.eps = eps
        self.std_eps = self.std + self.eps

    def forward(self, sample):
        signal = sample["signal"]

        if self.mean.get_device() != signal.get_device():
            self.mean = torch.as_tensor(self.mean, device=signal.device)
            self.std_eps = torch.as_tensor(self.std_eps, device=signal.device)

        signal.sub_(self.mean).div_(self.std_eps)
        sample["signal"] = signal
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


class EegChannelDropOut(torch.nn.Module):
    """DropOut some channels using the specified ratio.

    Args:
        p (float): Probability to drop each channel
    """

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, sample):
        signal = sample["signal"]
        ind = torch.rand((signal.shape[0], signal.shape[1]), device=signal.get_device()) < self.p
        signal[ind] = 0
        sample["signal"] = signal
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


class EegAdditiveGaussianNoise(torch.nn.Module):
    """Additive white Gaussian noise."""

    def __init__(self, mean=0.0, std=1e-2):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, sample):
        signal = sample["signal"]
        noise = torch.normal(
            mean=torch.ones_like(signal) * self.mean,
            std=torch.ones_like(signal) * self.std,
        )
        sample["signal"] = signal + noise
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


class EegMultiplicativeGaussianNoise(torch.nn.Module):
    """Multiplicative white Gaussian noise."""

    def __init__(self, mean=0.0, std=1e-2):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, sample):
        signal = sample["signal"]
        noise = torch.normal(
            mean=torch.ones_like(signal) * self.mean,
            std=torch.ones_like(signal) * self.std,
        )
        sample["signal"] = signal + (signal * noise)
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


class EegNormalizeAge(torch.nn.Module):
    """Normalize age of EEG metadata by the calculated statistics.

    Args:
        mean: Mean age of all people in EEG training dataset.
        std: Standard deviation of the age for all people in EEG training dataset.
        eps: Small number to prevent zero division.
    """

    def __init__(self, mean, std, eps=1e-8):
        super().__init__()
        self.mean = mean
        self.std = std
        self.eps = eps
        self.std_eps = self.std + self.eps

    def forward(self, sample):
        age = sample["age"]

        if not torch.is_tensor(self.mean) or self.mean.get_device() != age.get_device():
            self.mean = torch.as_tensor(self.mean, device=age.device)
            self.std_eps = torch.as_tensor(self.std_eps, device=age.device)

        age.sub_(self.mean).div_(self.std_eps)
        sample["age"] = age
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


class EegAddGaussianNoiseAge(torch.nn.Module):
    """Add a Gaussian noise to the age value

    Args:
        mean: Desired mean of noise level for the age value.
        std: Desired standard deviation of noise level for the age value.
    """

    def __init__(self, mean=0.0, std=1e-2):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, sample):
        age = sample["age"]
        noise = torch.normal(mean=torch.ones_like(age) * self.mean, std=torch.ones_like(age) * self.std)
        sample["age"] = age + noise
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


class EegAgeBias(torch.nn.Module):
    """Add a Gaussian noise to the age value

    Args:
        bias: Desired bias to add on age value.
    """

    def __init__(self, bias=0.0):
        super().__init__()
        self.bias = bias

    def forward(self, sample):
        sample["age"] += self.bias
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


class EegAgeSetConstant(torch.nn.Module):
    """Set the age value as the specified constant.

    Args:
        bias: Desired constant to set age value.
    """

    def __init__(self, bias=0.0):
        self.bias = bias
        super().__init__()

    def forward(self, sample):
        sample["age"] = torch.zeros_like(sample["age"]) + self.bias
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


class EegSpectrogram(torch.nn.Module):
    """Transform the multichannel 1D sequence as multichannel 2D image using short-time fourier transform
    (a.k.a. Spectrogram)"""

    def __init__(self, n_fft, complex_mode="as_real", **kwargs):
        super().__init__()
        if complex_mode not in ("as_real", "power", "remove"):
            raise ValueError('complex_mode must be set to one of ("as_real", "power", "remove")')

        self.n_fft = n_fft
        self.complex_mode = complex_mode
        self.stft_kwargs = kwargs

    def _spectrogram(self, x):
        if len(x.shape) == 3:
            N = x.shape[0]

            for i in range(N):
                xf = torch.stft(x[i], n_fft=self.n_fft, return_complex=True, **self.stft_kwargs)

                if i == 0:
                    if self.complex_mode == "as_real":
                        x_out = torch.zeros(
                            (N, 2 * xf.shape[0], xf.shape[1], xf.shape[2]),
                            dtype=x.dtype,
                            device=x.device,
                        )
                    else:
                        x_out = torch.zeros((N, *xf.shape), dtype=x.dtype, device=x.device)

                if self.complex_mode == "as_real":
                    x_out[i] = torch.cat(
                        (
                            torch.view_as_real(xf)[..., 0],
                            torch.view_as_real(xf)[..., 1],
                        ),
                        dim=0,
                    )
                elif self.complex_mode == "power":
                    x_out[i] = xf.abs()
                elif self.complex_mode == "remove":
                    x_out[i] = torch.real(xf)

        elif len(x.shape) == 2:
            xf = torch.stft(x, n_fft=self.n_fft, return_complex=True, **self.stft_kwargs)

            if self.complex_mode == "as_real":
                x_out = torch.cat(
                    (torch.view_as_real(xf)[..., 0], torch.view_as_real(xf)[..., 1]),
                    dim=0,
                )
            elif self.complex_mode == "power":
                x_out = xf.abs()
            elif self.complex_mode == "remove":
                x_out = torch.real(xf)

        else:
            raise ValueError(
                f'{self.__class__.__name__}._spectrogram(sample["signal"]) ' f"- check the signal tensor size."
            )

        return x_out

    def forward(self, sample):
        signal = sample["signal"]
        if torch.is_tensor(signal) is False:
            raise TypeError(
                "Before transforming the data signal as a spectrogram "
                "it must be converted to a PyTorch Tensor object using EegToTensor() transform."
            )

        sample["signal"] = self._spectrogram(signal)
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


class EegResample(torch.nn.Module):
    """Resample the EEG data as the specified sampling frequency.

    Args:
        orig_freq (int):
        new_freq (int):
        resampling_method (str, optional):
    """

    def __init__(
        self,
        orig_freq: int,
        new_freq: int,
        resampling_method: str = "sinc_interp_hann",
    ):
        super().__init__()
        if resampling_method not in (
            "sinc_interp_hann",
            "sinc_interp_kaiser",
        ):
            raise ValueError(
                f"{self.__class__.__name__}.__init__(resampling_method) must be set to one of "
                f'("sinc_interp_hann", "sinc_interp_kaiser")'
            )

        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.resampling_method = resampling_method

        if version.parse(torch.__version__) < version.parse("2.0.0"):
            if resampling_method == "sinc_interp_hann":
                resampling_method = "sinc_interpolation"
            elif resampling_method == "sinc_interp_kaiser":
                resampling_method = "kaiser_window"

        self.resampler = torchaudio.transforms.Resample(
            orig_freq=orig_freq,
            new_freq=new_freq,
            resampling_method=resampling_method,
        )

    def forward(self, sample):
        sample["signal"] = self.resampler(sample["signal"])
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


class TransformTimeChecker(object):
    def __init__(self, instance, header="", str_format=""):
        self.instance = instance
        self.header = header
        self.str_format = str_format

    def __call__(self, sample):
        start = time.time()
        sample = self.instance(sample)
        end = time.time()
        print(f"{self.header + type(self.instance).__name__:{self.str_format}}> {end - start :.5f}")
        return sample


def trim_trailing_zeros(a):
    assert type(a) == np.ndarray
    trim = 0
    for i in range(a.shape[-1]):
        if np.any(a[..., -1 - i] != 0):
            trim = i
            break
    if trim > 0:
        a = a[..., :-trim]
    return a
