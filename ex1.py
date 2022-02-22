from typing import Tuple
import numpy as np

Arr = np.ndarray


def prsep(c="=",n=50):
    print(c * n)


def fitness(x: Arr):
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))), axis=-1)


def update(
    v: Arr,
    x: Arr,
    x_best: Arr,
    best: Arr,
    omega: float,
    a1: float = 1.0,
    a2: float = 1.0,
    r1: float = 0.5,
    r2: float = 0.5,
) -> Tuple[Arr, Arr]:

    v_new = omega * v + a1 * r1 * (x_best - x) + a2 * r2 * (best - x)
    x_new = x + v_new

    fitt = fitness(x)
    fitt_new = fitness(x_new)

    fitt_stacked = np.stack([fitt, fitt_new])
    x_stacked = np.stack([x, x_new])

    x_best_new = x_stacked[np.argmin(fitt_stacked, axis=0),:,:]
    fitt_best_new = fitness(x_best_new)

    best_new = x_best_new[np.argmin(fitt_best_new)]

    return v_new, x_new, x_best_new, best_new


def main():
    xs = np.array([[-400, -400], [-410, -410], [-415, -415]], dtype=float)
    xs_best = xs.copy()
    best = xs[2, :]

    vs = np.array([[-50, -50]], dtype=float)

    fitnesses = fitness(xs)

    prsep()
    print("Ex. 1 a:")
    for i in range(3):
        print(f"Fitness x_{i+1}: {fitnesses[i]:.4f}")

    prsep()
    print("Ex. 1b:")
    for omega in [2.0, 0.5, 0.1]:
        prsep('-')
        print(f"{omega=}:")
        _, xs_new, *_ = update(vs, xs, xs_best, best, omega)
        fitt_new = fitness(xs_new)
        for i in range(3):
            print(f"\tParticle {i+1}: x={xs_new[i, :]}, f={fitt_new[i]}")



if __name__ == "__main__":
    main()
