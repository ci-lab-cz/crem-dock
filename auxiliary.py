from itertools import islice


def sort_two_lists(primary, secondary, reverse=False):
    # sort two lists by order of elements of the primary list
    paired_sorted = sorted(zip(primary, secondary), key=lambda x: x[0], reverse=reverse)
    return map(list, zip(*paired_sorted))  # two lists


def take(n, iterable):
    return list(islice(iterable, n))