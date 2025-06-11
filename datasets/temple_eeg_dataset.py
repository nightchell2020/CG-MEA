import os
from copy import deepcopy
import pyedflib

import numpy as np
import torch
from torch.utils.data import Dataset


class TuhAbnormalDataset(Dataset):
    """PyTorch Dataset Class for Temple University Hospital Abnormal Corpus v2.0.0.

    Args:
        data_list (list of dict): List of dictionary for the data.
        file_format (str): Determines which file format is used among of EDF, PyArrow Feather, and NumPy memmap.
        transform (callable): Optional transform to be applied on each data.
    """

    def __init__(self, data_list: list, file_format: str = "edf", transform=None):
        if file_format not in ["edf", "memmap"]:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(file_format) "
                f"must be set to one of 'edf', 'feather', 'memmap' and 'np'"
            )

        self.file_format = file_format
        self.transform = transform

        self.extension = file_format
        if file_format == "memmap":
            self.extension = "dat"

        self.signal_header = [
            "EEG FP1-REF",
            "EEG FP2-REF",
            "EEG F3-REF",
            "EEG F4-REF",
            "EEG C3-REF",
            "EEG C4-REF",
            "EEG P3-REF",
            "EEG P4-REF",
            "EEG O1-REF",
            "EEG O2-REF",
            "EEG F7-REF",
            "EEG F8-REF",
            "EEG T3-REF",
            "EEG T4-REF",
            "EEG T5-REF",
            "EEG T6-REF",
            "EEG A1-REF",
            "EEG A2-REF",
            "EEG FZ-REF",
            "EEG CZ-REF",
            "EEG PZ-REF",
            "EEG T1-REF",
            "EEG T2-REF",
            "EEG EKG1-REF",
        ]

        self.data_list = []
        for file in data_list:
            pathology = os.path.basename(os.path.dirname(file))
            self.data_list.append(
                {
                    "serial": os.path.basename(file),
                    "full_path": file,
                    "class_name": pathology,
                    "class_label": int(pathology == "abnormal"),
                }
            )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # annotation
        sample = deepcopy(self.data_list[idx])

        # signal
        sample["signal"] = self._read_signal(sample)
        sample["age"] = 0.0

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _read_signal(self, anno):
        if self.file_format == "edf":
            return self._read_edf(anno)
        else:
            return self._read_memmap(anno)

    def _read_edf(self, anno):
        signal, signal_headers, _ = pyedflib.highlevel.read_edf(anno["full_path"])
        return signal

    def _read_memmap(self, anno):
        signal = np.memmap(anno["full_path"], dtype="float32", mode="r").reshape(len(self.signal_header), -1)
        return signal
