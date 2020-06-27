import math
import logging
import numpy as np
import pandas as pd

from subgroup import Subgroup


class Beam:

    def __init__(self, subgroup: Subgroup, settings: dict):
        self.subgroups = [subgroup]
        self.candidates = []
        self.items = 1
        self.max_items = settings['width']
        try:
            self.candidate_size = int(settings['candidate_size'])
        except (KeyError, TypeError):
            self.candidate_size = settings['width'] ** 2
        self.strategy = settings['strategy']
        self.min_score = None
        self.scores = []

    def add(self, subgroup: Subgroup):
        if len(self.candidates) < self.candidate_size:
            self.candidates.append(subgroup)
            self.scores.append(subgroup.score)
            self.min_score = min(self.scores) if self.strategy == 'maximize' else max(self.scores)
        elif (self.strategy == 'maximize' and subgroup.score > self.min_score) or \
                (self.strategy == 'minimize' and subgroup.score < self.min_score):
            idx = self.scores.index(self.min_score)
            del self.scores[idx]
            del self.candidates[idx]
            self.candidates.append(subgroup)
            self.scores.append(subgroup.score)
            self.min_score = min(self.scores) if self.strategy == 'maximize' else max(self.scores)

    def sort(self, attribute: str = 'score') -> None:
        if attribute == 'score':
            self.candidates.sort(key=lambda x: x.score, reverse=(self.strategy == 'maximize'))
            self.subgroups.sort(key=lambda x: x.score, reverse=(self.strategy == 'maximize'))
        elif attribute == 'coverage':
            self.candidates.sort(
                key=lambda x: x.score * (x.coverage if (self.strategy == 'maximize') else (1 - x.coverage)),
                reverse=(self.strategy == 'maximize'))
        else:
            raise ValueError("Invalid sort attribute")

    def select_cover_based(self):
        self.sort()
        if self.candidate_size > self.max_items:
            index = np.array([])
            for subgroup in self.candidates:
                subgroup.coverage = 1 - (np.intersect1d(subgroup.data.index.values, index).size / subgroup.data.index.size)
                index = np.unique(np.concatenate((index, subgroup.data.index.values)))
            self.sort(attribute='coverage')
        self.subgroups = self.candidates[:self.max_items]
        self.scores = [s.score for s in self.subgroups]
        self.min_score = min(self.scores) if self.strategy == 'maximize' else max(self.scores)

    def decrypt_descriptions(self, translation):
        for s in self.subgroups:
            s.decrypt_description(translation)

    def print(self):
        self.sort(attribute='coverage')
        logging.debug("-" * 20)
        for s in self.subgroups:
            s.print()
