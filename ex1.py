from typing import Tuple
import numpy as np
from lib import SwarmOptimizer

Arr = np.ndarray


def prsep(c="=", n=50):
    print(c * n)


class Optimizer(SwarmOptimizer):
    def score(self, x: Arr):
        return np.sum(-x * np.sin(np.sqrt(np.abs(x))), axis=-1)


def main():
    xs = np.array([[-400, -400], [-410, -410], [-415, -415]], dtype=float)
    xs_best = xs.copy()
    best = xs[2, :]

    vs = np.array([[-50, -50]], dtype=float)

    hparams = dict(
        objective="max",
        a1=1.0,
        a2=1.0,
        r1=0.5,
        r2=0.5,
    )

    optim = Optimizer(**hparams)

    fitnesses = optim.score(xs)

    prsep()
    print("Ex. 1 a:")
    for i in range(3):
        print(f"Fitness x_{i+1}: {fitnesses[i]:.4f}")

    prsep()
    print("Ex. 1b:")
    for omega in [2.0, 0.5, 0.1]:
        prsep("-")
        print(f"{omega=}:")
        optim = Optimizer(omega=omega, **hparams)
        _, xs_new, *_ = optim.update_step(vs, xs, xs_best, best)
        fitt_new = optim.score(xs_new)
        for i in range(3):
            print(f"\tParticle {i+1}: x={xs_new[i, :]}, f={fitt_new[i]:.4f}")


if __name__ == "__main__":
    main()
