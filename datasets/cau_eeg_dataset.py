import os
import json
from copy import deepcopy
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
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


class CauEegDataset(Dataset):
    """PyTorch Dataset Class for CAU EEG Dataset.

    Args:
        root_dir (str): Root path to the EDF data files.
        metadata (list of dict): List of dictionary with metadata.
        load_event (bool): Determines whether to load event information or not for saving loading time.
        transform (callable): Optional transform to be applied on each data.
    """

    def __init__(self, root_dir, metadata, load_event, transform):
        self.root_dir = root_dir
        self.metadata = metadata
        self.load_event = load_event
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # metadata
        m = deepcopy(self.metadata[idx])

        # signal
        fname = os.path.join(self.root_dir, 'signal', m['serial'] + '.feather')
        df = pd.read_feather(fname)
        m['channel'] = df.columns.to_list()
        signal = df.to_numpy().T

        # event
        if self.load_event:
            fname = os.path.join(self.root_dir, 'event', m['serial'] + '.json')
            with open(fname, 'r') as json_file:
                m['event'] = json.load(json_file)

        # pack into dictionary
        sample = {'signal': signal,
                  'age': m['age'],
                  'class_label': m['class_label'],
                  'metadata': m}

        if self.transform:
            sample = self.transform(sample)

        return sample

