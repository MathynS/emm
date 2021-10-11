import logging
import pandas as pd

from copy import deepcopy
from typing import List, Optional, Union
from multiprocessing import Process, cpu_count

from beam import Beam
from util import downsize, is_notebook
from subgroup import Subgroup
from evaluation_metrics import EvaluationMetric, Correlation, cleanup
from visualization import Visualize
from description import Description
from workers import create_subgroups, evaluate_subgroups, beam_adder
import multiproc

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# logger = logging.getLogger(__name__)
evaluate_queue = multiproc.Queue()
add_queue = multiproc.Queue()


class EMM:

    def __init__(self, width: int, depth: int,
                 evaluation_metric: EvaluationMetric, n_jobs: int = -1,
                 strategy: str = 'maximize', n_bins: int = 10,
                 bin_strategy: Optional[str] = 'equidepth',
                 candidate_size: int = None, log_level=50):
        logging.basicConfig(filename=None, level=log_level,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.depth = depth
        self.evaluation_metric = evaluation_metric
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = min(n_jobs, cpu_count())
        print(f"Running with {self.n_jobs} "
              f"job{'s' if self.n_jobs > 1 else ''}...")
        self.settings = dict(
            strategy=strategy,
            width=width,
            n_bins=n_bins,
            bin_strategy=bin_strategy,
            candidate_size=candidate_size
        )
        self.beam = None
        self.target_columns = None
        self.dataset_target = None
        self.dataset = None

    def __setattr__(self, name, value):
        if name in ['depth', 'n_jobs'] and not isinstance(value, int):
            raise TypeError(f"Invalid type for setting {name}: only integers "
                            f"are allowed")
        if name == 'settings':
            if not isinstance(value['width'], int):
                raise TypeError("Invalid type for setting width: only "
                                "integers are allowed")
            if value['strategy'] not in ['minimize', 'maximize']:
                raise ValueError(
                    f"Invalid option for strategy: {value['strategy']}, "
                    f"allowed options are `maximize` and `minimize`")
            if not isinstance(value['n_bins'], int):
                raise TypeError(f"Invalid type for setting {value['n_bins']}: "
                                f"only integers are allowed")
            if not isinstance(value['candidate_size'], int) \
                    and value['candidate_size'] is not None:
                raise TypeError(f"Invalid type for setting {value['n_bins']}: "
                                f"only integers or None are allowed")
            if not value['bin_strategy'] in ['equiwidth', 'equidepth', None]:
                raise ValueError(f"Invalid option for bin_strategy: "
                                 f"{value['bin_strategy']}, allowed options"
                                 f" are 'equidepth', 'equiwidth', and None")
        super().__setattr__(name, value)

    def search(self, data: pd.DataFrame, target_cols: Union[List[str], str],
               descriptive_cols: List[str] = None):
        logging.info("Start")
        data, translations = downsize(deepcopy(data))
        self.settings['object_cols'] = translations
        self.dataset = Subgroup(data, Description('all'))
        _, self.dataset.target = self.evaluation_metric.calculate(
            data[target_cols], data[target_cols]
        )
        self.beam = Beam(self.dataset, self.settings)
        target_cols = list(target_cols,)
        if descriptive_cols is None:
            descriptive_cols = [c for c in data.columns if c not in target_cols]
        elif any(c in descriptive_cols for c in target_cols):
            raise ValueError("The target and descriptive columns may not "
                             "overlap!")
        if any(c not in data.columns for c in descriptive_cols + target_cols):
            raise ValueError("All specified columns should be present in the "
                             "dataset")
        self.dataset_target = data[target_cols]
        self.target_columns = target_cols
        for _ in tqdm(range(self.depth)):
            self.make_subgroups(descriptive_cols)
            self.depth -= 1
            self.beam.select_cover_based()

        self.beam.decrypt_descriptions(translations)
        self.beam.print()
        cleanup()

    def visualize(self, subgroups: int = None, cols: int = 3, height: int = 460,
                  include_dataset: bool =True):
        """Calls the appropriate visualization function.

        Args:
            subgroups: Number of subgroups to visualize.
            cols: Number of columns to display.
            height: Height of each plot in pixels.
            include_dataset: Show the full dataset as its own plot. Shows it if
                true.
        """
        subgroups = len(self.beam.subgroups) if subgroups is None else subgroups
        vis = Visualize(eval_metric=self.evaluation_metric,
                        dataset=self.dataset,
                        subgroups=self.beam.subgroups,
                        target_columns=self.target_columns,
                        translations=self.settings['object_cols'],
                        cols=cols,
                        group_size=subgroups,
                        include_dataset=include_dataset,
                        height=height)
        vis.visualize()

    def make_subgroups(self, cols: List[str]):
        beam_workers = []
        for _ in range(self.n_jobs):
            w = Process(target=evaluate_subgroups,
                        args=(evaluate_queue, add_queue, self.target_columns,
                              self.dataset_target,
                              self.evaluation_metric.calculate))
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


if __name__ == "__main__":
    # DEBUG Debugging code to test EMM with Housing
    df = pd.read_csv('../example/data/Mini-Housing.csv')
    clf = EMM(width=40, depth=1, evaluation_metric=Correlation(),
              n_jobs=1)

    clf.search(df, target_cols=['price', 'lotsize'])

    clf.visualize(subgroups=5, cols=3)
