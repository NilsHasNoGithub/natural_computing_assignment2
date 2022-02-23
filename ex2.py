from cProfile import label
import numpy as np
from lib import SwarmOptimizer, Arr
import click
import os
import matplotlib.pyplot as plt

class Optimizer(SwarmOptimizer):

    def score(self, x: Arr):
        return x ** 2


@click.command()
@click.option("--result-dir", default="results", help="Directory to save results to.")
def main(result_dir: str):
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

    results_1 = optim_1.optimize(v, x)
    results_2 = optim_2.optimize(v, x)

    traj_1 = [s[0,0] for s in results_1.x_hist]
    traj_2 = [s[0,0] for s in results_2.x_hist]

    scores_1 = [s[0,0] for s in results_1.score_hist]
    scores_2 = [s[0,0] for s in results_2.score_hist]

    x_min = min(traj_1 + traj_2)
    x_max = max(traj_1 + traj_2)

    plt.figure(figsize=(10, 5))
    plt.title("Particle trajectory")

    for i, (traj, scores, omega) in enumerate(zip([traj_1, traj_2], [scores_1, scores_2], [0.5, 0.7])):
        plt.subplot(1, 2, i + 1)
        plt.title(f"Omega = {omega}")
        plt.plot(traj, scores)
        plt.scatter([traj[0]],[scores[0]], label="start", color="green")
        plt.scatter([traj[-1]],[scores[-1]], label="end", color="red")
        # plt.plot(x_range, [optim_1.score(np.array([[x]]))[0,0] for x in x_range])
        plt.xlabel("x")
        plt.ylabel("$x^2$")
    
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(result_dir, "loss_2b.png"))

if __name__ == "__main__":
    main()