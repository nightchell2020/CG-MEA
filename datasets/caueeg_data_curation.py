from dataclasses import dataclass, asdict, field
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np


@dataclass
class MultiEegLabel:
    """Dataclass for EEG multi-property diagnosis label."""

    dementia: bool = field(default=False, metadata={"help": "Dementia."})
    ad: bool = field(default=False, metadata={"help": "Alzheimer's disease dementia."})
    load: bool = field(default=False, metadata={"help": "Late-onset AD."})
    eoad: bool = field(default=False, metadata={"help": "Early-onset AD."})
    vd: bool = field(default=False, metadata={"help": "Vascular dementia."})
    sivd: bool = field(
        default=False, metadata={"help": "Subcortical ischemic vascular dementia."}
    )
    ad_vd_mixed: bool = field(
        default=False,
        metadata={
            "help": "Mixed dementia of Alzheimer's disease " "and vascular dementia."
        },
    )

    mci: bool = field(default=False, metadata={"help": "Mild cognitive impairment."})
    mci_ad: bool = field(
        default=False, metadata={"help": "MCI with amyloid PET positive."}
    )
    mci_amnestic: bool = field(default=False, metadata={"help": "Amnestic MCI."})
    mci_amnestic_ef: bool = field(
        default=False, metadata={"help": "MCI encoding failure."}
    )
    mci_amnestic_rf: bool = field(
        default=False, metadata={"help": "MCI retrieval failure."}
    )
    mci_non_amnestic: bool = field(
        default=False, metadata={"help": "Non-amnestic MCI."}
    )
    mci_multi_domain: bool = field(
        default=False, metadata={"help": "Multi-domain MCI."}
    )
    mci_vascular: bool = field(default=False, metadata={"help": "Vascular MCI."})

    normal: bool = field(default=False, metadata={"help": "Normal."})
    cb_normal: bool = field(default=False, metadata={"help": "Community-based normal."})
    smi: bool = field(
        default=False,
        metadata={
            "help": "Subjective memory impairment (subjective cognitive decline)."
        },
    )
    hc_normal: bool = field(
        default=False, metadata={"help": "Health care center normal."}
    )

    ftd: bool = field(default=False, metadata={"help": "Frontotemporal dementia."})
    bvftd: bool = field(default=False, metadata={"help": "Behavioral variant FTD."})
    language_ftd: bool = field(
        default=False, metadata={"help": "Language variant FTD."}
    )
    semantic_aphasia: bool = field(
        default=False, metadata={"help": "Semantic aphasia."}
    )
    non_fluent_aphasia: bool = field(
        default=False, metadata={"help": "Non-fluent aphasia."}
    )

    parkinson_synd: bool = field(
        default=False, metadata={"help": "Parkinson's syndrome."}
    )
    parkinson_disease: bool = field(
        default=False, metadata={"help": "Parkinson's disease."}
    )
    parkinson_dementia: bool = field(
        default=False, metadata={"help": "Parkinson's disease dementia."}
    )

    nph: bool = field(
        default=False, metadata={"help": "Normal pressure hydrocephalus."}
    )
    tga: bool = field(default=False, metadata={"help": "Transient global amnesia."})

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(
                    f"ERROR: MultiEegLabel.__init__() has unknown label: {k}"
                )
            setattr(self, k, v)

    def __repr__(self):
        return self.__class__.__qualname__ + str(
            {k: v for k, v in asdict(self).items() if v is True}
        )

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
        assert (
            type(dx1) == str
        ), f"ERROR: load_from_string function input is non-string type: {type(dx1)}"

        label = MultiEegLabel()

        if dx1 in ["load"]:
            label = MultiEegLabel(dementia=True, ad=True, load=True)
        elif dx1 in ["eoad"]:
            label = MultiEegLabel(dementia=True, ad=True, eoad=True)

        elif dx1 in ["vd", "vascular dementia", "sivd"]:
            label = MultiEegLabel(dementia=True, vd=True, sivd=True)
        elif dx1 in ["ad-vd-mixed"]:
            label = MultiEegLabel(dementia=True, ad_vd_mixed=True)

        elif dx1 in ["mci"]:
            label = MultiEegLabel(mci=True)

        elif dx1 in ["ad_mci", "ad-mci"]:
            label = MultiEegLabel(mci=True, mci_ad=True)
        elif dx1 in ["ad-mci amnestic"]:
            label = MultiEegLabel(mci=True, mci_ad=True, mci_amnestic=True)

        elif dx1 in ["ad-mci (ef)"]:
            label = MultiEegLabel(
                mci=True, mci_ad=True, mci_amnestic=True, mci_amnestic_ef=True
            )
        elif dx1 in ["ad-mci (rf)"]:
            label = MultiEegLabel(
                mci=True, mci_ad=True, mci_amnestic=True, mci_amnestic_rf=True
            )

        elif dx1 in ["mci amnestic", "amci"]:
            label = MultiEegLabel(mci=True, mci_amnestic=True)
        elif dx1 in ["mci amnestic multi-domain"]:
            label = MultiEegLabel(mci=True, mci_amnestic=True, mci_multi_domain=True)

        elif dx1 in [
            "mci_ef",
            "mci ef",
            "mci(ef)",
            "amci (ef)",
            "amci(ef)",
            "mci encoding failure",
        ]:
            label = MultiEegLabel(mci=True, mci_amnestic=True, mci_amnestic_ef=True)
        elif dx1 in ["mci (ef) multi-domain", "mci encoding failure multi-domain"]:
            label = MultiEegLabel(
                mci=True, mci_amnestic=True, mci_amnestic_ef=True, mci_multi_domain=True
            )

        elif dx1 in [
            "mci_rf",
            "mci rf",
            "mci (rf)",
            "amci rf",
            "mci retrieval failure",
        ]:
            label = MultiEegLabel(mci=True, mci_amnestic=True, mci_amnestic_rf=True)
        elif dx1 in ["mci(rf) multi-domain"]:
            label = MultiEegLabel(
                mci=True, mci_amnestic=True, mci_amnestic_rf=True, mci_multi_domain=True
            )

        elif dx1 in ["mci non amnestic", "mci non-amnestic", "cind"]:
            label = MultiEegLabel(mci=True, mci_non_amnestic=True)

        elif dx1 in ["vascular mci"]:
            label = MultiEegLabel(mci=True, mci_vascular=True)
        elif dx1 in ["vmci non-amnestic"]:
            label = MultiEegLabel(mci=True, mci_vascular=True, mci_non_amnestic=True)
        elif dx1 in ["vmci(ef)"]:
            label = MultiEegLabel(
                mci=True, mci_amnestic=True, mci_amnestic_ef=True, mci_vascular=True
            )
        elif dx1 in ["vmci(rf)", "vascular mci (rf)"]:
            label = MultiEegLabel(
                mci=True, mci_amnestic=True, mci_amnestic_rf=True, mci_vascular=True
            )

        elif dx1 in ["nc", "nl"]:
            label = MultiEegLabel(normal=True)
        elif dx1 in ["cb_normal"]:
            label = MultiEegLabel(normal=True, cb_normal=True)
        elif dx1 in ["smi"]:
            label = MultiEegLabel(normal=True, smi=True)
        elif dx1 in ["hc_normal"]:
            label = MultiEegLabel(normal=True, hc_normal=True)

        elif dx1 in ["ftd"]:
            label = MultiEegLabel(ftd=True)
        elif dx1 in ["bvftd"]:
            label = MultiEegLabel(ftd=True, bvftd=True)
        elif dx1 in ["language ftd"]:
            label = MultiEegLabel(ftd=True, language_ftd=True)
        elif dx1 in ["semantic aphasia"]:
            label = MultiEegLabel(ftd=True, semantic_aphasia=True)
        elif dx1 in ["non fluent aphasia"]:
            label = MultiEegLabel(ftd=True, non_fluent_aphasia=True)

        elif dx1 in ["parkinson_synd", "other parkinson synd"]:
            label = MultiEegLabel(parkinson_synd=True)
        elif dx1 in ["pd", "parkinson's disease"]:
            label = MultiEegLabel(parkinson_synd=True, parkinson_disease=True)
        elif dx1 in ["pdd", "parkinson dementia"]:
            label = MultiEegLabel(
                dementia=True, parkinson_synd=True, parkinson_dementia=True
            )

        elif dx1 in ["nph"]:
            label = MultiEegLabel(nph=True)

        elif dx1 in ["tga"]:
            label = MultiEegLabel(tga=True)

        else:
            if dx1 in ["unknown", "0", "?검사없음"]:
                label = MultiEegLabel()
            else:
                print(
                    f"(Warning) load_from_string function cannot parse the input: {dx1}"
                )

        assert "label" in dir(), "load_from_string() - unknown dx1 label: %s" % dx1
        return label


def birth_to_datetime(b):
    try:
        if b is None:
            return None
        elif type(b) is int:
            y = (b // 10000) + 1900
            m = (b % 10000) // 100
            d = b % 100
            return datetime.date(y, m, d)
        elif type(b) is str:
            b = int(b)
            y = (b // 10000) + 1900
            m = (b % 10000) // 100
            d = b % 100
            return datetime.date(y, m, d)
    except Exception as e:
        print(
            f"WARNING - Input to birth_to_datetime() is uninterpretable: {e}, {type(b)}, {b}"
        )
    return None


def calculate_age(birth, record):
    if birth is None:
        return None
    try:
        age = (
            record - relativedelta(years=birth.year, months=birth.month, days=birth.day)
        ).year
        if age < 40 or 100 < age:
            print(f"WARNING - calculate_age() generated an unordinary age: {age}")
        return age
    except Exception as e:
        print(f"WARNING - calculate_age() has an exception: {e}")
    return None


def serialize_json(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, datetime.datetime) or isinstance(obj, datetime.date):
        serial = obj.isoformat()
        return serial

    if isinstance(obj, MultiEegLabel):
        serial = obj.get_true_keys()
        return serial

    return obj.__dict__
