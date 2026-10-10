"""NumPy K-means and silhouette calculation for AE latent vectors."""

from typing import Dict, List, Tuple

import numpy as np


def kmeans_plus_plus(
    values: np.ndarray, n_clusters: int, rng: np.random.Generator
) -> np.ndarray:
    """Initialize KMeans centers with the standard k-means++ rule."""
    count = len(values)
    selected: List[int] = [int(rng.integers(count))]
    while len(selected) < n_clusters:
        squared = np.sum(
            (values[:, None, :] - values[np.asarray(selected)][None, :, :]) ** 2,
            axis=2,
        )
        closest = squared.min(axis=1)
        closest[np.asarray(selected)] = 0.0
        total = float(closest.sum())
        if total <= 1e-15:
            remaining = [index for index in range(count) if index not in selected]
            selected.append(remaining[0])
        else:
            selected.append(int(rng.choice(count, p=closest / total)))
    return values[np.asarray(selected)].copy()


def kmeans_once(
    values: np.ndarray,
    n_clusters: int,
    seed: int,
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    rng = np.random.default_rng(seed)
    centers = kmeans_plus_plus(values, n_clusters, rng)
    previous = None
    for iteration in range(1, max_iter + 1):
        squared = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(squared, axis=1)
        if previous is not None and np.array_equal(labels, previous):
            break
        previous = labels.copy()
        new_centers = centers.copy()
        closest = squared[np.arange(len(values)), labels]
        for cluster in range(n_clusters):
            members = values[labels == cluster]
            if len(members):
                new_centers[cluster] = members.mean(axis=0)
            else:
                replacement = int(np.argmax(closest))
                new_centers[cluster] = values[replacement]
                closest[replacement] = -np.inf
        centers = new_centers
    squared = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(squared, axis=1)
    inertia = float(squared[np.arange(len(values)), labels].sum())
    return labels.astype(int), centers, inertia, iteration


def fit_kmeans(
    values: np.ndarray,
    n_clusters: int,
    seed: int,
    n_init: int,
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray, float, int, int]:
    """Choose the minimum-inertia solution across deterministic initializations."""
    best = None
    for initialization in range(n_init):
        # Consecutive seeds mirror a conventional multi-start KMeans run and
        # make independent base seeds heavily overlap when n_init is large.
        # This avoids reporting a rare local minimum selected by one unusual
        # seed stride as the formal solution.
        run_seed = int(seed + initialization)
        labels, centers, inertia, iterations = kmeans_once(
            values, n_clusters, run_seed, max_iter
        )
        signature = tuple(labels.tolist())
        candidate = (inertia, signature, labels, centers, iterations, run_seed)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    return best[2], best[3], float(best[0]), int(best[4]), int(best[5])


def distance_metrics(distance: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    upper_i, upper_j = np.triu_indices_from(distance, k=1)
    same = labels[upper_i] == labels[upper_j]
    within = float(distance[upper_i[same], upper_j[same]].mean())
    between = float(distance[upper_i[~same], upper_j[~same]].mean())
    silhouettes = sample_silhouettes(distance, labels)
    return {
        "silhouette": float(silhouettes.mean()),
        "mean_within_distance": within,
        "mean_between_distance": between,
        "between_within_ratio": between / max(within, 1e-15),
    }


def euclidean_distance_matrix(values: np.ndarray) -> np.ndarray:
    delta = values[:, None, :] - values[None, :, :]
    matrix = np.sqrt(np.sum(delta * delta, axis=2))
    np.fill_diagonal(matrix, 0.0)
    return matrix


def sample_silhouettes(distance: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.zeros(len(labels), dtype=float)
    for index, cluster in enumerate(labels):
        same = np.flatnonzero(labels == cluster)
        same = same[same != index]
        if same.size == 0:
            values[index] = 0.0
            continue
        within = float(np.mean(distance[index, same]))
        alternatives = [
            float(np.mean(distance[index, labels == other]))
            for other in np.unique(labels)
            if other != cluster
        ]
        between = min(alternatives)
        denominator = max(within, between)
        values[index] = 0.0 if denominator <= 1e-15 else (between - within) / denominator
    return values
