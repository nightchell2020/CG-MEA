import os
import json
from copy import deepcopy
from dataclasses import dataclass, asdict
import pyedflib

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import torch
from torch.utils.data import Dataset


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

        elif dx1 in ['parkinson_synd', 'other parkinson synd']:
            label = MultiLabel(parkinson_synd=True)
        elif dx1 in ['pd', 'parkinson\'s disease']:
            label = MultiLabel(parkinson_synd=True, parkinson_disease=True)
        elif dx1 in ['pdd', 'parkinson dementia']:
            label = MultiLabel(dementia=True, parkinson_synd=True, parkinson_dementia=True)

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


class CauEegDataset(Dataset):
    """PyTorch Dataset Class for CAUEEG Dataset.

    Args:
        root_dir (str): Root path to the EDF data files.
        data_list (list of dict): List of dictionary for the data.
        load_event (bool): Determines whether to load event information or not for saving loading time.
        file_format (str): Determines which file format is used among of EDF, PyArrow Feather, and NumPy memmap.
        transform (callable): Optional transform to be applied on each data.
    """

    def __init__(self, root_dir: str, data_list: list,
                 load_event: bool, file_format: str = 'edf', transform=None):
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

        # metadata
        anno = deepcopy(self.data_list[idx])

        # signal
        if self.file_format == 'edf':
            signal = self.read_edf(anno)
        elif self.file_format == 'feather':
            signal = self.read_feather(anno)
        else:
            signal = self.read_memmap(anno)

        # event
        if self.load_event:
            anno['event'] = self.read_event(anno)

        # pack into dictionary
        sample = {'signal': signal,
                  **anno}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def read_edf(self, m):
        edf_file = os.path.join(self.root_dir, 'signal', m['serial'] + '.edf')
        signal, signal_headers, _ = pyedflib.highlevel.read_edf(edf_file)
        # m['channel_header'] = [s_h['label'] for s_h in signal_headers]
        return signal

    def read_feather(self, m):
        fname = os.path.join(self.root_dir, 'signal/feather', m['serial'] + '.feather')
        df = feather.read_feather(fname)
        # m['channel_header'] = df.columns.to_list()
        return df.values.T

    def read_memmap(self, m):
        fname = os.path.join(self.root_dir, 'signal/memmap', m['serial'] + '.dat')
        signal = np.memmap(fname, dtype='int32', mode='r').reshape(21, -1)
        return signal

    def read_np(self, m):
        fname = os.path.join(self.root_dir, 'signal', m['serial'] + '.npy')
        return np.load(fname)

    # def read_hdf5(self, m):
    #     return self.f_handle[m['serial']][:]

    # def read_jay(self, m):
    #     fname = os.path.join(self.root_dir, 'signal', m['serial'] + '.jay')
    #     signal = dt.fread(fname).to_numpy().T
    #     return signal

    # def read_parquet(self, m):
    #     fname = os.path.join(self.root_dir, 'signal', m['serial'] + '.parquet')
    #     signal = pd.read_parquet(fname).values.T
    #     return signal

    def read_event(self, m):
        fname = os.path.join(self.root_dir, 'event', m['serial'] + '.json')
        with open(fname, 'r') as json_file:
            event = json.load(json_file)
        return event

    def get_data_frame(self, idx=0):
        m = self.data_list[idx]
        fname = os.path.join(self.root_dir, 'signal/feather', m['serial'] + '.feather')
        df = pd.read_feather(fname)
        return df
