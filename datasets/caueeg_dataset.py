import os
import json
from copy import deepcopy
from dataclasses import dataclass, asdict, field
import pyedflib

import numpy as np
import pyarrow.feather as feather
import torch
from torch.utils.data import Dataset


@dataclass
class MultiEegLabel:
    """Dataclass for EEG multi-property diagnosis label.
    """
    dementia: bool = field(default=False, metadata={"help": "Dementia."})
    ad: bool = field(default=False, metadata={"help": "Alzheimer's disease dementia."})
    load: bool = field(default=False, metadata={"help": "Late-onset AD."})
    eoad: bool = field(default=False, metadata={"help": "Early-onset AD."})
    vd: bool = field(default=False, metadata={"help": "Vascular dementia."})
    sivd: bool = field(default=False, metadata={"help": "Subcortical ischemic vascular dementia."})
    ad_vd_mixed: bool = field(default=False, metadata={"help": "Mixed dementia of Alzheimer's disease "
                                                               "and vascular dementia."})

    mci: bool = field(default=False, metadata={"help": "Mild cognitive impairment."})
    mci_ad: bool = field(default=False, metadata={"help": "MCI with amyloid PET positive."})
    mci_amnestic: bool = field(default=False, metadata={"help": "Amnestic MCI."})
    mci_amnestic_ef: bool = field(default=False, metadata={"help": "MCI encoding failure."})
    mci_amnestic_rf: bool = field(default=False, metadata={"help": "MCI retrieval failure."})
    mci_non_amnestic: bool = field(default=False, metadata={"help": "Non-amnestic MCI."})
    mci_multi_domain: bool = field(default=False, metadata={"help": "Multi-domain MCI."})
    mci_vascular: bool = field(default=False, metadata={"help": "Vascular MCI."})

    normal: bool = field(default=False, metadata={"help": "Normal."})
    cb_normal: bool = field(default=False, metadata={"help": "Community-based normal."})
    smi: bool = field(default=False, metadata={"help": "Subjective memory impairment (subjective cognitive decline)."})
    hc_normal: bool = field(default=False, metadata={"help": "Health care center normal."})

    ftd: bool = field(default=False, metadata={"help": "Frontotemporal dementia."})
    bvftd: bool = field(default=False, metadata={"help": "Behavioral variant FTD."})
    language_ftd: bool = field(default=False, metadata={"help": "Language variant FTD."})
    semantic_aphasia: bool = field(default=False, metadata={"help": "Semantic aphasia."})
    non_fluent_aphasia: bool = field(default=False, metadata={"help": "Non-fluent aphasia."})

    parkinson_synd: bool = field(default=False, metadata={"help": "Parkinson's syndrome."})
    parkinson_disease: bool = field(default=False, metadata={"help": "Parkinson's disease."})
    parkinson_dementia: bool = field(default=False, metadata={"help": "Parkinson's disease dementia."})

    nph: bool = field(default=False, metadata={"help": "Normal pressure hydrocephalus."})
    tga: bool = field(default=False, metadata={"help": "Transient global amnesia."})

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f'ERROR: MultiEegLabel.__init__() has unknown label: {k}')
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

        label = MultiEegLabel()

        if dx1 in ['load']:
            label = MultiEegLabel(dementia=True, ad=True, load=True)
        elif dx1 in ['eoad']:
            label = MultiEegLabel(dementia=True, ad=True, eoad=True)

        elif dx1 in ['vd', 'vascular dementia', 'sivd']:
            label = MultiEegLabel(dementia=True, vd=True, sivd=True)
        elif dx1 in ['ad-vd-mixed']:
            label = MultiEegLabel(dementia=True, ad_vd_mixed=True)

        elif dx1 in ['mci']:
            label = MultiEegLabel(mci=True)

        elif dx1 in ['ad_mci', 'ad-mci']:
            label = MultiEegLabel(mci=True, mci_ad=True)
        elif dx1 in ['ad-mci amnestic']:
            label = MultiEegLabel(mci=True, mci_ad=True, mci_amnestic=True)

        elif dx1 in ['ad-mci (ef)']:
            label = MultiEegLabel(mci=True, mci_ad=True, mci_amnestic=True, mci_amnestic_ef=True)
        elif dx1 in ['ad-mci (rf)']:
            label = MultiEegLabel(mci=True, mci_ad=True, mci_amnestic=True, mci_amnestic_rf=True)

        elif dx1 in ['mci amnestic', 'amci']:
            label = MultiEegLabel(mci=True, mci_amnestic=True)
        elif dx1 in ['mci amnestic multi-domain']:
            label = MultiEegLabel(mci=True, mci_amnestic=True, mci_multi_domain=True)

        elif dx1 in ['mci_ef', 'mci ef', 'mci(ef)', 'amci (ef)', 'amci(ef)', 'mci encoding failure']:
            label = MultiEegLabel(mci=True, mci_amnestic=True, mci_amnestic_ef=True)
        elif dx1 in ['mci (ef) multi-domain', 'mci encoding failure multi-domain']:
            label = MultiEegLabel(mci=True, mci_amnestic=True, mci_amnestic_ef=True, mci_multi_domain=True)

        elif dx1 in ['mci_rf', 'mci rf', 'mci (rf)', 'amci rf', 'mci retrieval failure']:
            label = MultiEegLabel(mci=True, mci_amnestic=True, mci_amnestic_rf=True)
        elif dx1 in ['mci(rf) multi-domain']:
            label = MultiEegLabel(mci=True, mci_amnestic=True, mci_amnestic_rf=True, mci_multi_domain=True)

        elif dx1 in ['mci non amnestic', 'mci non-amnestic', 'cind']:
            label = MultiEegLabel(mci=True, mci_non_amnestic=True)

        elif dx1 in ['vascular mci']:
            label = MultiEegLabel(mci=True, mci_vascular=True)
        elif dx1 in ['vmci non-amnestic']:
            label = MultiEegLabel(mci=True, mci_vascular=True, mci_non_amnestic=True)
        elif dx1 in ['vmci(ef)']:
            label = MultiEegLabel(mci=True, mci_amnestic=True, mci_amnestic_ef=True, mci_vascular=True)
        elif dx1 in ['vmci(rf)', 'vascular mci (rf)']:
            label = MultiEegLabel(mci=True, mci_amnestic=True, mci_amnestic_rf=True, mci_vascular=True)

        elif dx1 in ['nc', 'nl']:
            label = MultiEegLabel(normal=True)
        elif dx1 in ['cb_normal']:
            label = MultiEegLabel(normal=True, cb_normal=True)
        elif dx1 in ['smi']:
            label = MultiEegLabel(normal=True, smi=True)
        elif dx1 in ['hc_normal']:
            label = MultiEegLabel(normal=True, hc_normal=True)

        elif dx1 in ['ftd']:
            label = MultiEegLabel(ftd=True)
        elif dx1 in ['bvftd']:
            label = MultiEegLabel(ftd=True, bvftd=True)
        elif dx1 in ['language ftd']:
            label = MultiEegLabel(ftd=True, language_ftd=True)
        elif dx1 in ['semantic aphasia']:
            label = MultiEegLabel(ftd=True, semantic_aphasia=True)
        elif dx1 in ['non fluent aphasia']:
            label = MultiEegLabel(ftd=True, non_fluent_aphasia=True)

        elif dx1 in ['parkinson_synd', 'other parkinson synd']:
            label = MultiEegLabel(parkinson_synd=True)
        elif dx1 in ['pd', 'parkinson\'s disease']:
            label = MultiEegLabel(parkinson_synd=True, parkinson_disease=True)
        elif dx1 in ['pdd', 'parkinson dementia']:
            label = MultiEegLabel(dementia=True, parkinson_synd=True, parkinson_dementia=True)

        elif dx1 in ['nph']:
            label = MultiEegLabel(nph=True)

        elif dx1 in ['tga']:
            label = MultiEegLabel(tga=True)

        else:
            if dx1 in ['unknown', '0', '?검사없음']:
                label = MultiEegLabel()
            else:
                print(f'(Warning) load_from_string function cannot parse the input: {dx1}')

        assert 'label' in dir(), 'load_from_string() - unknown dx1 label: %s' % dx1
        return label


class CauEegDataset(Dataset):
    """PyTorch Dataset Class for CAUEEG Dataset.

    Args:
        root_dir (str): Root path to the EDF data files.
        data_list (list of dict): List of dictionary for the data.
        load_event (bool): Determines whether to load event information or not for saving loading time.
        file_format (str): Determines which file format is used among of EDF, PyArrow Feather, and NumPy memmap.
        transform (callable): Optional transform to be applied on each data.
    """

    def __init__(self, root_dir: str, data_list: list, load_event: bool,
                 file_format: str = 'edf', transform=None):
        if file_format not in ['edf', 'feather', 'memmap', 'np']:
            raise ValueError(f"{self.__class__.__name__}.__init__(file_format) "
                             f"must be set to one of 'edf', 'feather', 'memmap' and 'np'")

        self.root_dir = root_dir
        self.data_list = data_list
        self.load_event = load_event
        self.file_format = file_format
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # annotation
        sample = deepcopy(self.data_list[idx])

        # signal
        sample['signal'] = self._read_signal(sample)

        # event
        if self.load_event:
            sample['event'] = self._read_event(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _read_signal(self, anno):
        if self.file_format == 'edf':
            return self._read_edf(anno)
        elif self.file_format == 'feather':
            return self._read_feather(anno)
        else:
            return self._read_memmap(anno)

    def _read_edf(self, anno):
        edf_file = os.path.join(self.root_dir, f"signal/{anno['serial']}.edf")
        signal, signal_headers, _ = pyedflib.highlevel.read_edf(edf_file)
        return signal

    def _read_feather(self, anno):
        fname = os.path.join(self.root_dir, f"signal/feather/{anno['serial']}.feather")
        df = feather.read_feather(fname)
        return df.values.T

    def _read_memmap(self, anno):
        fname = os.path.join(self.root_dir, f"signal/memmap/{anno['serial']}.dat")
        signal = np.memmap(fname, dtype='int32', mode='r').reshape(21, -1)
        return signal

    def _read_np(self, anno):
        fname = os.path.join(self.root_dir, f"signal/{anno['serial']}.npy")
        return np.load(fname)

    def _read_event(self, m):
        fname = os.path.join(self.root_dir, 'event', m['serial'] + '.json')
        with open(fname, 'r') as json_file:
            event = json.load(json_file)
        return event
