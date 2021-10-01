import time
import pandas as pd

from copy import deepcopy
from multiprocessing import Queue

from subgroup import Subgroup


def create_subgroups(subgroup: Subgroup, column: str, queue: Queue, settings: dict):
    if column in subgroup.description:
        return
    data = subgroup.data
    values = list(data[column].unique())
    if len(values) == 1:  # No need to make a split for a single value
        return
    if column in settings['object_cols'] or len(values) < settings['n_bins']:
        while len(values) > 0:
            if queue.qsize() < 10:  # Reasonable size to keep in the beam
                value = values.pop(0)
                subset = data[data[column] == value]
                queue.put(Subgroup(subset, deepcopy(subgroup.description).extend(column, value)))
            else:
                time.sleep(.1)  # Else try again in a .1 second
    else:  # Float or Int
        if settings['bin_strategy'] == 'equidepth':
            _, intervals = pd.qcut(data[column].tolist(), q=min(settings['n_bins'], len(values)),
                                   duplicates='drop', retbins=True)
        else:
            raise ValueError(f"Invalid bin strategy `{settings['strategy']}`")
        intervals = list(intervals)
        lower_bound = intervals.pop(0)
        while len(intervals) > 0:
            if queue.qsize() < 10:
                upper_bound = intervals.pop(0)
                subset = data[(data[column] > lower_bound) & (data[column] <= upper_bound)]
                queue.put(Subgroup(subset, deepcopy(subgroup.description).extend(column, [lower_bound, upper_bound])))
                lower_bound = upper_bound
            else:
                time.sleep(.1)  # Else try again in a .1 second


def evaluate_subgroups(queue_from, queue_to, target_columns, dataset_target, score):
    while True:
        item = queue_from.get()
        if item == 'done':
            queue_to.put('done')
            break
        if len(item.data[target_columns]) == 0:
            continue
        subgroup_target = item.data[target_columns]
        item.score, item.target = score(subgroup_target, dataset_target)
        item.print()
        queue_to.put(item)


def beam_adder(queue, beam, n_jobs):
    workers = n_jobs
    while workers > 0:
        item = queue.get()
        if item == 'done':
            workers -= 1
            continue
        beam.add(item)
