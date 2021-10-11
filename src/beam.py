import logging
import numpy as np

from subgroup import Subgroup


class Beam:
    def __init__(self, subgroup: Subgroup, settings: dict):
        """Beam class to perform beam search on subgroups.

        Reference implementation of Algorithm 1 from "Exceptional Model Mining"
        by W. Duivesteijn (2016).

        Args:
            subgroup: The dataset or subgroup for which a beam search should be
                performed on.
            settings: Accepts the following keys: "width", "candidate_size",
                "strategy". "width" is an int, "candidate_size" is an int and
                defaults to "width" ** 2, "strategy" is a string of either
                "maximize" or "minimize".
        """
        self.subgroups = [subgroup]
        self.candidates = []
        self.items = 1

        # Make sure that candidate size has a value. It may sometimes be None
        # or not exist in which case a default value is assigned.
        try:
            if settings['candidate_size'] is None:
                settings['candidate_size'] = settings['width'] ** 2
        except KeyError:
            settings['candidate_size'] = settings['width'] ** 2

        self.max_items = settings['width']
        self.candidate_size = int(settings['candidate_size'])
        self.strategy = settings['strategy']
        self.min_score = None
        self.scores = []

    def add(self, subgroup: Subgroup):
        def update():
            """Made its own function because this is reused in both branches."""
            self.candidates.append(subgroup)
            self.scores.append(subgroup.score)
            if self.strategy == "maximize":
                self.min_score = min(self.scores)
            else:
                self.min_score = max(self.scores)

        if len(self.candidates) < self.candidate_size:
            update()
        elif (self.strategy == 'maximize'
              and subgroup.score > self.min_score) or \
                (self.strategy == 'minimize'
                 and subgroup.score < self.min_score):
            idx = self.scores.index(self.min_score)
            del self.scores[idx]
            del self.candidates[idx]
            update()

    def sort(self, attribute: str = 'score') -> None:
        if attribute == 'score':
            self.candidates.sort(key=lambda x: x.score,
                                 reverse=(self.strategy == 'maximize'))
            self.subgroups.sort(key=lambda x: x.score,
                                reverse=(self.strategy == 'maximize'))
        elif attribute == 'coverage':
            self.candidates.sort(
                key=lambda x: x.score * (x.coverage
                                         if (self.strategy == 'maximize')
                                         else (1 - x.coverage)),
                reverse=(self.strategy == 'maximize'))
        else:
            raise ValueError("Invalid sort attribute")

    def select_cover_based(self):
        self.sort()
        if self.candidate_size > self.max_items:
            index = np.array([])
            for subgroup in self.candidates:
                subgroup.coverage = 1 - (
                        np.intersect1d(subgroup.data.index.values, index).size
                        / subgroup.data.index.size)
                index = np.unique(
                    np.concatenate((index, subgroup.data.index.values)))
            self.sort(attribute='coverage')
        self.subgroups = self.candidates[:self.max_items]
        self.scores = [s.score for s in self.subgroups]

        if self.strategy == "maximize":
            self.min_score = min(self.scores)
        else:
            self.min_score = max(self.scores)

    def decrypt_descriptions(self, translation):
        for s in self.subgroups:
            s.decrypt_description(translation)

    def print(self):
        self.sort(attribute='coverage')
        logging.debug("-" * 20)
        for s in self.subgroups:
            s.print()
