from abc import ABC, abstractmethod
from typing import List
import numpy as np
from dataclasses import dataclass

from tqdm import tqdm

Arr = np.ndarray

@dataclass
class OptimizeResults:
    x_hist: List[Arr]
    v_hist: List[Arr]
    score_hist: List[Arr]


class SwarmOptimizer(ABC):
    def __init__(
        self,
        omega: float = 0.5,
        a1: float = 1.0,
        a2: float = 1.0,
        r1: float = 0.5,
        r2: float = 0.5,
        objective="max",
    ) -> None:
        super().__init__()

        self._maximize = objective == "max"

        if objective not in ["max", "min"]:
            raise ValueError(
                f"Objective must be either 'max' or 'min', not {objective}"
            )

        self._omega = omega
        self._a1 = a1
        self._a2 = a2
        self._r1 = r1
        self._r2 = r2

    @abstractmethod
    def score(self, x: Arr):
        ...

    def update_step(
        self,
        v: Arr,
        x: Arr,
        x_best: Arr,
        best: Arr,
    ):
        arg_best = np.argmax if self._maximize else np.argmin

        v_new = (
            self._omega * v
            + self._a1 * self._r1 * (x_best - x)
            + self._a2 * self._r2 * (best - x)
        )
        x_new = x + v_new

        fitt = self.score(x)
        fitt_new = self.score(x_new)

        fitt_stacked = np.stack([fitt, fitt_new])
        x_stacked = np.stack([x, x_new])

        x_best_new = np.zeros_like(x_best)
        idxs_best = arg_best(fitt_stacked, axis=0)

        for i, idx in enumerate(idxs_best):
            x_best_new[i, :] = x_stacked[idx, i, :]

        fitt_best_new = self.score(x_best_new)

        best_new = x_best_new[arg_best(fitt_best_new)]

        return v_new, x_new, x_best_new, best_new

    def optimize(self, v: Arr, x: Arr, n_steps: int = 100):
        arg_best = np.argmax if self._maximize else np.argmin

        score = self.score(x)
        x_best = x.copy()
        best = x[arg_best(score)]

        x_hist = [x]
        v_hist = [v]
        score_hist = [score]

        for _ in tqdm(range(n_steps)):
            v, x, x_best, best = self.update_step(v, x, x_best, best)

            x_hist.append(x)
            v_hist.append(v)
            score_hist.append(self.score(x))

        return OptimizeResults(x_hist, v_hist, score_hist)

