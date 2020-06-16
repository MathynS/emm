import pandas as pd

from typing import Any, List, Optional, Union
from multiprocessing import Manager, Process, Queue, cpu_count

from beam import Beam
from util import downsize
from subgroup import Subgroup
from evaluation_metrics import metrics
from description import Description
from workers import create_subgroups, evaluate_subgroups, beam_adder


evaluate_queue = Queue()
add_queue = Queue()


class EMM:

    def __init__(self, width: int, depth: int, evaluation_metric: Union[str, callable], n_jobs: int = -1,
                 strategy: str = 'maximize', n_bins: int = 10, bin_strategy: Optional[str] = 'equidepth',
                 candidate_size: int = None):
        if not isinstance(width, int):
            raise ValueError("Invalid type for setting width: only integers are allowed")
        if isinstance(depth, int):
            self.depth = depth
        else:
            raise ValueError("Invalid type for setting depth: only integers are allowed")
        if not isinstance(n_jobs, int):
            raise ValueError("Invalid type for setting n_jobs: only integers are allowed")
        elif n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = min(n_jobs, cpu_count())
        if hasattr(evaluation_metric, '__call__'):
            self.evaluation_metric = evaluation_metric
        else:
            try:
                self.evaluation_metric = metrics[evaluation_metric]
            except KeyError:
                raise ValueError(f"Nu such metric: {evaluation_metric}")
        self.settings = dict(
            strategy=strategy,
            width=width,
            n_bins=n_bins,
            bin_strategy=bin_strategy,
            candidate_size=candidate_size
        )
        if strategy not in ['minimize', 'maximize']:
            raise ValueError(f"Invalid option for strategy {strategy}, allowed options are `maximize` and `minimize`")
        self.beam = None
        self.target_columns = None
        self.dataset_target = None
        self.dataset = None

    def search(self, data: pd.DataFrame, target_cols: Union[List[str], str], descriptive_cols: List[str] = None):
        print("Start")
        data, translations = downsize(data)
        self.settings['object_cols'] = translations
        self.dataset = Subgroup(data, Description('all'))
        self.beam = Beam(self.dataset, self.settings)
        target_cols = list(target_cols,)
        if descriptive_cols is None:
            descriptive_cols = [c for c in data.columns if c not in target_cols]
        elif any(c in descriptive_cols for c in target_cols):
            raise ValueError("The target and descriptive columns may not overlap!")
        if any(c not in data.columns for c in descriptive_cols + target_cols):
            raise ValueError("All specified columns should be present in the dataset")
        self.dataset_target = data[target_cols]
        self.target_columns = target_cols
        while self.depth > 0:
            self.make_subgroups(descriptive_cols)
            self.depth -= 1
            self.beam.select_cover_based()
            print("-" * 10)
        self.beam.decrypt_descriptions(translations)
        self.beam.print()

    def make_subgroups(self, cols: List[str]):
        beam_workers = []
        for _ in range(self.n_jobs):
            w = Process(target=evaluate_subgroups, args=(evaluate_queue, add_queue, self.target_columns,
                                                         self.dataset_target, self.evaluation_metric))
            w.start()
            beam_workers.append(w)
        for subgroup in self.beam.subgroups:
            for col in cols:
                create_subgroups(subgroup, col, evaluate_queue, self.settings)
        for _ in range(self.n_jobs):
            evaluate_queue.put('done')
        beam_adder(add_queue, self.beam, self.n_jobs)
        for w in beam_workers:
            w.join()
