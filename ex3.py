from email.policy import default
import itertools
from typing import Callable, Tuple, Union
import click
import numpy as np

from lib import Arr, SwarmOptimizer
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder


def k_means(data: Arr, k: int, max_iter: int = 100):
    n_samples, n_features = data.shape
    data = data.copy()

    centroids = data[np.random.choice(n_samples, k, replace=False), :]

    for _ in range(max_iter):
        closest_centroids = []

        for i in range(n_samples):
            distances = np.linalg.norm(data[i, :] - centroids, axis=-1)
            closest_centroids.append(np.argmin(distances))

        closest_centroids = np.array(closest_centroids)

        for i in range(k):
            centroids[i, :] = np.mean(data[closest_centroids == i, :], axis=0)

    return centroids, closest_centroids


def quantization_error(centroids, closest_centroids, data):
    sum_ = 0.0
    for i in range(len(closest_centroids)):
        sum_ += np.linalg.norm(centroids[closest_centroids[i]] - data[i])

    return sum_


def test_k_means(data: Arr, n_classes: int, n_inits: int=10, max_iter:int=100) -> Tuple[Arr, float]:
    all_centroids = []
    errors = []

    for _ in range(n_inits):
        centroids, closest_centroids = k_means(data, n_classes, max_iter=max_iter)
        all_centroids.append(centroids)
        errors.append(quantization_error(centroids, closest_centroids, data))

    best_idx = np.argmin(errors)

    return all_centroids[best_idx], errors[best_idx]


def calc_closest_centroids(data, centroids):
    n_samples, _ = data.shape
    closest_centroids = []

    for i in range(n_samples):
        distances = np.linalg.norm(data[i, :] - centroids, axis=-1)
        closest_centroids.append(np.argmin(distances))

    return closest_centroids


def calc_score(particle, data):
    closest_centroids = calc_closest_centroids(data, particle)
    return quantization_error(particle, closest_centroids, data)


class Optimzer(SwarmOptimizer):
    def __init__(
        self,
        data: Arr,
        omega: float = 0.5,
        a1: float = 1,
        a2: float = 1,
        r1: Union[float, Callable] = 0.5,
        r2: Union[float, Callable] = 0.5,
        objective="max",
    ) -> None:
        super().__init__(omega, a1, a2, r1, r2, objective)
        self._data = data

    def score(self, x: Arr):
        result = np.zeros(x.shape[0])

        for i in range(x.shape[0]):
            result[i] = calc_score(x[i, :, :], self._data)

        return result

def test_poc(data: Arr, n_clusters: int, n_particles: int, max_iter: int, omega: float, a1: float, a2:float):
    n_samples, n_features = data.shape

    r = lambda: np.random.uniform(0, 1, size=(1, n_features))
    optim = Optimzer(data, objective="min", omega=omega, a1=a1, a2=a1, r1 = r, r2 = r)

    particles = [
        data[np.random.choice(n_samples, n_clusters, replace=False), :]
        for _ in range(n_particles)
    ]

    velocities = [np.zeros_like(particle) for particle in particles]

    particles = np.stack(particles)
    velocities = np.stack(velocities)

    results = optim.optimize(velocities, particles, n_steps=max_iter, use_tqdm=False)

    return results.best, optim.score(np.array([results.best]))[0]


def artificial_data_1(n_samples=400) -> Tuple[Arr, Arr]:
    samples = np.random.uniform(-1, 1, size=(n_samples, 2))

    labels = (samples[:, 0] >= 0.7) | (
        (samples[:, 0] <= 0.3) & (samples[:, 1] >= (-0.2 - samples[:, 0]))
    )

    return samples, labels


def plot_iris_clusters(iris_data: Arr, labels: Arr, title: str) -> "Figure":
    n_samples, n_features = iris_data.shape
    n_plots = n_features ** 2

    assert n_features == 4


    fig, subplots = plt.subplots(n_features, n_features, figsize=(13, 13))
    fig.suptitle(title)

    for i, j in itertools.product(range(n_features), range(n_features)):
        p = subplots[i, j]
        if i == j:
            p.hist(iris_data[:, i], bins=20)
            p.set_xlabel(f"$x_{i}$")
        else:
            p.scatter(iris_data[:, i], iris_data[:, j], c=labels, cmap="tab10")
            p.set_xlabel(f"$x_{i}$")
            p.set_ylabel(f"$x_{j}$")

    fig.tight_layout()
    return fig  

def print_err_result(typ: str, n_rounds: int, err: float):
    print(f"Quatization error for {typ}, averaged over {n_rounds} runs: {err:.4f}")

@click.command()
@click.option(
    "--data-file", "-d", type=click.Path(exists=True), default="data/iris.data"
)
@click.option("--result-dir", "-r", type=click.Path(), default="results")
@click.option("--n-rounds", "-n", type=click.INT, default=30)
def main(data_file: str, result_dir: str, n_rounds: int):
    os.makedirs(result_dir, exist_ok=True)

    n_clusters = 3

    iris_df = pd.read_csv(data_file, header=None)
    iris_data = iris_df.iloc[:, :-1].values
    iris_labels = iris_df.iloc[:, -1].to_list()

    k_means_results = Parallel(n_jobs=-1)(delayed(test_k_means)(iris_data, n_clusters) for _ in range(n_rounds))
    k_means_error = np.mean([s[1] for s in k_means_results])

    k_means_result_iris = k_means_results[0][0]

    print_err_result("k-means", n_rounds, k_means_error)

    n_particles = 10
    max_iter = 100

    omega = 0.72
    a1 = 1.49
    a2 = 1.49

    # quantitization error over n_rounds runs:
    results = Parallel(n_jobs=-1)(
        delayed(test_poc)(iris_data, n_clusters, n_particles, max_iter, omega, a1, a2)
        for _ in range(n_rounds)
    )
    scores = [s[1] for s in results]

    poc_results_iris = results[0][0]

    print_err_result("POC", n_rounds, np.mean(scores))

    # same for artificila data

    n_clusters = 2
    n_particles = 10
    max_iter = 100

    artificial_data, labels = artificial_data_1()

    k_means_results = Parallel(n_jobs=-1)(delayed(test_k_means)(artificial_data, n_clusters) for _ in range(n_rounds))
    k_means_error = np.mean([s[1] for s in k_means_results])

    k_means_centroids_artificial = k_means_results[0][0]

    print_err_result("k-means", n_rounds, k_means_error)

    results = Parallel(n_jobs=-1)(
        delayed(test_poc)(
            artificial_data, n_clusters, n_particles, max_iter, omega, a1, a2
        )
        for _ in range(n_rounds)
    )
    scores = [s[1] for s in results]

    poc_centroids_artificial = results[0][0]

    print_err_result("POC", n_rounds, np.mean(scores))

    # plot results for artificial data
    fig, subplots = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Artificial data clusters")

    subplots[0].scatter(
        artificial_data[:, 0], artificial_data[:, 1], c=labels, cmap="tab10"
    )
    subplots[0].set_title("True labels")

    k_means_labels = calc_closest_centroids(
        artificial_data, k_means_centroids_artificial
    )
    subplots[1].scatter(
        artificial_data[:, 0], artificial_data[:, 1], c=k_means_labels, cmap="tab10"
    )
    subplots[1].set_title("K-means labels")

    poc_labels = calc_closest_centroids(artificial_data, poc_centroids_artificial)
    subplots[2].scatter(
        artificial_data[:, 0], artificial_data[:, 1], c=poc_labels, cmap="tab10"
    )
    subplots[2].set_title("POC labels")

    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, "ex3_artificial_data_1.png"))


    # plot iris results

    # kmeans
    labels = calc_closest_centroids(iris_data, k_means_result_iris)
    fig = plot_iris_clusters(iris_data, labels, "Iris data clusters generated with k-means")
    fig.savefig(os.path.join(result_dir, "ex3_iris_data_kmeans.png"))

    # poc
    labels = calc_closest_centroids(iris_data, poc_results_iris)
    fig = plot_iris_clusters(iris_data, labels, "Iris data clusters generated with POC")
    fig.savefig(os.path.join(result_dir, "ex3_iris_data_poc.png"))

    # ground truth
    labels = LabelEncoder().fit_transform(iris_labels)
    fig = plot_iris_clusters(iris_data, labels, "Iris data clusters true labels")
    fig.savefig(os.path.join(result_dir, "ex3_iris_data_ground_truth.png"))



if __name__ == "__main__":
    main()
