import os
from multiprocessing import cpu_count


def cpu_type(x):
    return max(1, min(int(x), cpu_count()))


def filepath_type(x):
    if x:
        return os.path.abspath(x)
    else:
        return x


def str_lower_type(x):
    if x:
        return x.lower()
    else:
        return x


def similarity_value_type(x):
    return max(0, min(1, float(x)))


def ring_size_type(x):
    if x is None:
        return None
    if len(x) not in (1, 2):
        raise ValueError('--ring_size accepts either one value (fixed ring size) or two values (min max window), '
                         f'got {len(x)}: {x}')
    if any(v < 3 for v in x):
        raise ValueError(f'--ring_size values must be at least 3 (a ring cannot have fewer atoms), got {x}')
    if len(x) == 1:
        return x[0]
    return tuple(sorted(x))