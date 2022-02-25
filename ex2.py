from cProfile import label
import numpy as np
from lib import SwarmOptimizer, Arr
import click
import os
import matplotlib.pyplot as plt


class Optimizer(SwarmOptimizer):
    def score(self, x: Arr):
        return x**2


@click.command()
@click.option("--result-dir", default="results", help="Directory to save results to.")
@click.option(
    "--num-steps", default=100, help="Number of steps to run the optimizer for."
)
def main(result_dir: str, num_steps: int):
    os.makedirs(result_dir, exist_ok=True)

    x = np.array([[20]])
    v = np.array([[10]])

    optim_1 = Optimizer(
        objective="min",
        omega=0.5,
        a1=1.5,
        a2=1.5,
        r1=0.5,
        r2=0.5,
    )

    optim_2 = Optimizer(
        objective="min",
        omega=0.7,
        a1=1.5,
        a2=1.5,
        r1=1.0,
        r2=1.0,
    )

    results_1 = optim_1.optimize(v, x, n_steps=num_steps)
    results_2 = optim_2.optimize(v, x, n_steps=num_steps)

    traj_1 = [s[0, 0] for s in results_1.x_hist]
    traj_2 = [s[0, 0] for s in results_2.x_hist]

    scores_1 = [s[0, 0] for s in results_1.score_hist]
    scores_2 = [s[0, 0] for s in results_2.score_hist]

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(f"Particle trajectory, for {num_steps} steps")

    for i, (traj, scores) in enumerate(zip([traj_1, traj_2], [scores_1, scores_2])):
        plt.subplot(1, 2, i + 1)
        plt.title(f"Configuration {i + 1}")
        plt.plot(traj, scores)
        plt.scatter([traj[0]], [scores[0]], label="start", color="green", s=150)
        plt.scatter([traj[-1]], [scores[-1]], label="end", color="red", s=150)
        # plt.plot(x_range, [optim_1.score(np.array([[x]]))[0,0] for x in x_range])
        plt.xlabel("x")
        plt.ylabel("$x^2$")

    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(result_dir, f"trajectory_2b_steps_{num_steps}.png"))


if __name__ == "__main__":
    main()
