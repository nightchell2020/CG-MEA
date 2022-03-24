import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import time

from .cau_eeg_dataset import MultiLabel


def trim_tailing_zeros(a):
    assert type(a) == np.ndarray
    trim = 0
    for i in range(a.shape[-1]):
        if np.any(a[..., -1 - i] != 0):
            trim = i
            break
    a = a[..., :-trim]
    return a


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
        print(f'WARNING - Input to birth_to_datetime() is uninterpretable: {e}, {type(b)}, {b}')
    return None


def calculate_age(birth, record):
    if birth is None:
        return None
    try:
        age = (record - relativedelta(years=birth.year, months=birth.month, days=birth.day)).year
        if age < 40 or 100 < age:
            print(f'WARNING - calculate_age() generated an unordinary age: {age}')
        return age
    except Exception as e:
        print(f'WARNING - calculate_age() has an exception: {e}')
    return None


def serialize_json(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, datetime.datetime) or isinstance(obj, datetime.date):
        serial = obj.isoformat()
        return serial

    if isinstance(obj, MultiLabel):
        serial = obj.get_true_keys()
        return serial

    return obj.__dict__


class TransformTimeChecker(object):
    def __init__(self, instance, header='', str_format=''):
        self.instance = instance
        self.header = header
        self.str_format = str_format

    def __call__(self, sample):
        start = time.time()
        sample = self.instance(sample)
        end = time.time()
        print(f'{self.header + type(self.instance).__name__:{self.str_format}}> {end - start :.5f}')
        return sample
