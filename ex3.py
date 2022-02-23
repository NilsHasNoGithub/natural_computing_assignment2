from email.policy import default
from typing import Tuple
import click
import numpy as np
from tqdm import tqdm

from lib import Arr
from copy import deepcopy
import pandas as pd
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import os


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


def test_k_means(data, n_classes) -> Tuple[Arr, float]:
    centroids, closest_centroids = k_means(data, n_classes, max_iter=100)
    return centroids, quantization_error(centroids, closest_centroids, data)


def calc_closest_centroids(data, centroids):
    n_samples, n_features = data.shape
    closest_centroids = []

    for i in range(n_samples):
        distances = np.linalg.norm(data[i, :] - centroids, axis=-1)
        closest_centroids.append(np.argmin(distances))

    return closest_centroids


def calc_score(particle, data):
    closest_centroids = calc_closest_centroids(data, particle)
    return quantization_error(particle, closest_centroids, data)


def test_poc(data, n_clusters, n_particles, max_iter, omega, a1, a2):
    n_samples, n_features = data.shape

    particles = [
        data[np.random.choice(n_samples, n_clusters, replace=False), :]
        for _ in range(n_particles)
    ]
    particles_best = deepcopy(particles)

    velocities = [np.zeros_like(particle) for particle in particles]

    particle_best_scores = [calc_score(p, data) for p in particles_best]
    global_best = particles[np.argmin([calc_score(p, data) for p in particles])]
    global_best_score = calc_score(global_best, data)

    for _ in range(max_iter):

        particle_scores = [calc_score(p, data) for p in particles]

        for i in range(n_particles):
            if particle_scores[i] < particle_best_scores[i]:
                particles_best[i] = particles[i]
                particle_best_scores[i] = particle_scores[i]

        best_idx = np.argmin(particle_best_scores)

        if particle_best_scores[best_idx] < global_best_score:
            global_best = particles_best[best_idx]
            global_best_score = particle_best_scores[best_idx]

        for i in range(n_particles):
            velocities[i] = (
                omega * velocities[i]
                + a1
                * np.random.uniform(0, 1, size=(1, n_features))
                * (particles_best[i] - particles[i])
                + a2
                * np.random.uniform(0, 1, size=(1, n_features))
                * (global_best - particles[i])
            )
            particles[i] += velocities[i]

        # print(velocities[0])
        # print(global_best_score)

    return global_best, global_best_score


def artificial_data_1(n_samples=400) -> Tuple[Arr, Arr]:
    samples = np.random.uniform(-1, 1, size=(n_samples, 2))

    labels = (samples[:, 0] >= 0.7 )| ((samples[:, 0] <= 0.3) & (samples[:, 1] >= (-0.2 - samples[:, 0])))

    return samples, labels


@click.command()
@click.option(
    "--data-file", "-d", type=click.Path(exists=True), default="data/iris.data"
)
@click.option("--result-dir", "-r", type=click.Path(), default="results")
def main(data_file: str, result_dir: str):
    os.makedirs(result_dir, exist_ok=True)

    n_clusters = 3
    n_rounds = 1

    iris_data = pd.read_csv(data_file, header=None).iloc[:, :-1].values

    k_means_results = [test_k_means(iris_data, n_clusters) for _ in range(n_rounds)]
    k_means_error = np.mean([s[1] for s in k_means_results])    

    k_means_result_iris = k_means_results[0][0]

    print(f"Quantization error k_means, averaged over {n_rounds} runs: {k_means_error}")

    n_particles = 10
    max_iter = 100

    omega = 0.72
    a1 = 1.49
    a2 = 1.49

    # quantitization error over n_rounds runs:
    results = Parallel(n_jobs=cpu_count())(
        delayed(test_poc)(iris_data, n_clusters, n_particles, max_iter, omega, a1, a2)
        for _ in tqdm(range(n_rounds))
    )
    scores = [s[1] for s in results]

    poc_results_iris = results[0][0]

    print(f"Quantization error POC, averaged over {n_rounds} runs: {np.mean(scores)}")

    # same for artificila data

    n_clusters = 2
    n_particles = 10
    max_iter = 100

    artificial_data, labels = artificial_data_1()

    k_means_results = [test_k_means(artificial_data, n_clusters) for _ in range(n_rounds)]
    k_means_error = np.mean([s[1] for s in k_means_results])   

    k_means_centroids_artificial = k_means_results[0][0] 

    print(f"Quantization error k_means averaged over {n_rounds} runs: {k_means_error}")

    results = Parallel(n_jobs=cpu_count())(
        delayed(test_poc)(artificial_data, n_clusters, n_particles, max_iter, omega, a1, a2)
        for _ in tqdm(range(n_rounds))
    )
    scores = [s[1] for s in results]

    poc_centroids_artificial = results[0][0]

    print(f"Quantization error POC for artificial data, averaged over {n_rounds} runs: {np.mean(scores)}")

    # plot results for artificial data
    fig, subplots = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Artificial data clusters")

    subplots[0].scatter(artificial_data[:, 0], artificial_data[:, 1], c=labels, cmap="tab10")
    subplots[0].set_title("True labels")

    k_means_labels = calc_closest_centroids(artificial_data, k_means_centroids_artificial)
    subplots[1].scatter(artificial_data[:, 0], artificial_data[:, 1], c=k_means_labels, cmap="tab10")
    subplots[1].set_title("K-means labels")

    poc_labels = calc_closest_centroids(artificial_data, poc_centroids_artificial)
    subplots[2].scatter(artificial_data[:, 0], artificial_data[:, 1], c=poc_labels, cmap="tab10")
    subplots[2].set_title("POC labels")

    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, "ex3_artificial_data_1.png"))




if __name__ == "__main__":
    main()
