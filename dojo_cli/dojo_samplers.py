import random
import time
from dataclasses import replace
from typing import Callable, List, Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from pacman_env import MazeSpec

# ============================================================
# 🧩 Shared Utilities
# ============================================================


def _random_pellets(
    base_spec: MazeSpec,
    pellet_density: Optional[float],
    pellet_count: Optional[int],
) -> List[Tuple[int, int]]:
    """
    Generate a random pellet configuration based on pellet_mode and density.
    Always returns at least one pellet.
    """
    w, h = base_spec.width, base_spec.height
    coords = [(x, y) for x in range(w) for y in range(h)]

    if base_spec.pellet_mode == "single":
        # One pellet, random position
        pellet_positions = [random.choice(coords)]
    elif pellet_density is not None:
        n_pellets = max(1, int(pellet_density * w * h))
        pellet_positions = random.sample(coords, n_pellets)
    else:
        # Fallback to base_spec default
        pellet_positions = base_spec.pellet_positions

    if pellet_count is not None:
        pellet_positions = pellet_positions[:pellet_count]

    return pellet_positions


def _render_ascii(spec: MazeSpec) -> str:
    """Render MazeSpec as a square ASCII maze."""
    w, h = spec.width, spec.height
    grid = [[" " for _ in range(w)] for _ in range(h)]

    for x, y in spec.pellet_positions:
        if 0 <= x < w and 0 <= y < h:
            grid[y][x] = "•"
            print("placed pellet at:", x, y)

    px, py = spec.pacman_start
    if 0 <= px < w and 0 <= py < h:
        grid[py][px] = "P"
        print("placed pacman at:", px, py)

    horizontal = "─" * w
    lines = ["┌" + horizontal + "┐"]
    for row in grid:
        lines.append("│" + "".join(row) + "│")
    lines.append("└" + horizontal + "┘")
    return "\n".join(lines)


# ============================================================
# 🎲 Random Episode Sampler
# ============================================================


def random_episode_sampler(
    base_spec: MazeSpec,
    *,
    randomize_pacman: bool = True,
    randomize_single_target: bool = True,
    pellet_density: Optional[float] = None,
    pellet_count: Optional[int] = None,
    **kwargs,
) -> Callable[[], MazeSpec]:
    """
    Return a callable sampler that produces randomized MazeSpecs.
    Compatible with all existing cluster or training code.
    """
    coords = [(x, y) for x in range(base_spec.width) for y in range(base_spec.height)]

    def sampler() -> MazeSpec:
        pacman_start = random.choice(coords) if randomize_pacman else base_spec.pacman_start
        pellet_positions = base_spec.pellet_positions

        if base_spec.pellet_mode == "single":
            if randomize_single_target:
                candidates = [pos for pos in coords if pos != pacman_start]
                pellet_positions = [random.choice(candidates)] if candidates else [pacman_start]
        elif pellet_density is not None:
            pellet_positions = _random_pellets(base_spec, pellet_density, pellet_count)

        if pellet_count is not None:
            pellet_positions = pellet_positions[:pellet_count]

        return replace(
            base_spec,
            pacman_start=pacman_start,
            pellet_positions=pellet_positions,
            random_seed=random.randint(0, 10**9),
        )

    return sampler


# ============================================================
# 🧮 Cluster Sampler (original behavior, unchanged)
# ============================================================


def cluster_sampler(
    base_spec: MazeSpec,
    n_clusters=128,
    n_samples=1_000_000,
    pellet_density=None,
    pellet_count=None,  # count of pellets
    **kwargs,
):
    """Sample random mazes, cluster them by features, and return a callable sampler."""
    maze_specs = []
    features = []
    start = time.time()

    print(f"cluster_sampler: generating {n_samples} samples...")

    for _ in range(n_samples):

        spec = random_episode_sampler(base_spec, pellet_density=pellet_density, pellet_count=pellet_count, **kwargs)()
        if pellet_count is not None:
            spec = replace(spec, pellet_positions=spec.pellet_positions[:pellet_count])

        if spec.pellet_positions:
            xs = [x for x, y in spec.pellet_positions]
            ys = [y for x, y in spec.pellet_positions]
            mean_x, mean_y = np.mean(xs), np.mean(ys)
            var_x, var_y = np.var(xs), np.var(ys)
        else:
            mean_x = mean_y = 0.0
            var_x = var_y = 0.0

        f = [
            spec.width * spec.height,
            len(spec.pellet_positions) / (spec.width * spec.height),
            mean_x,
            mean_y,
            var_x,
            var_y,
            spec.pacman_start[0],
            spec.pacman_start[1],
        ]
        features.append(f)
        maze_specs.append(spec)
    print(f"generated samples in {time.time() - start:.2f}s")

    print("clustering samples...")
    start = time.time()
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000, n_init="auto")
    labels = kmeans.fit_predict(features)

    clusters = {i: [] for i in range(n_clusters)}
    for spec, label in zip(maze_specs, labels):
        clusters[label].append(spec)

    clusters = {k: v for k, v in clusters.items() if v}
    cluster_keys = list(clusters.keys())
    idx = 0
    print("total clusters with samples:", len(cluster_keys))
    print("clustering done in {:.2f}s".format(time.time() - start))

    # --- visualize a few random cluster samples ---
    # print(f"Previewing {min(10, len(clusters))} cluster samples before continuing:")
    # for i, (key, specs) in enumerate(clusters.items()):
    #     if i >= 10:
    #         break
    #     print(f"\nCluster {key} ({len(specs)} samples)")
    #     for _ in range(5):
    #         sample_spec = random.choice(specs)
    #         ascii_output = _render_ascii(sample_spec)
    #         print(ascii_output)
    #     input("Press Enter to view next cluster...")

    # print("✅ Cluster preview complete.")

    def sampler():
        """Cycle through clusters and return random sample from each (skipping empty ones)."""
        nonlocal idx
        attempts = 0
        while attempts < len(cluster_keys):
            key = cluster_keys[idx % len(cluster_keys)]
            idx += 1
            if clusters[key]:
                return random.choice(clusters[key])
            attempts += 1
        print("⚠️ All clusters empty — falling back to uniform random spec")
        return random.choice(maze_specs)

    return sampler


# ============================================================
# from sklearn.cluster import MiniBatchKMeans


def lattice_cluster_sampler(
    base_spec: MazeSpec,
    n_clusters=128,
    pellet_density=None,
    total_samples=1_000_000,
    pellet_count=None,
    **kwargs,
):
    """
    Generate a large balanced dataset of MazeSpecs covering spatial diversity,
    then cluster them to provide structured curriculum sampling.

    Args:
        base_spec (MazeSpec): Base environment specification.
        n_clusters (int): Number of KMeans clusters.
        pellet_density (float): Optional pellet density for randomization.
        total_samples (int): Total MazeSpecs to generate (default 1M).
        **kwargs:
            per_start_samples (int): Overrides total_samples; legacy.
    """
    w, h = base_spec.width, base_spec.height
    coords = [(x, y) for x in range(w) for y in range(h)]
    n_positions = len(coords)

    # --- Determine samples per start position ---
    per_start = kwargs.get("per_start_samples")
    if per_start is None:
        per_start = max(1, total_samples // n_positions)

    print(f"Generating lattice samples ({n_positions} positions × {per_start} samples each)...")
    maze_specs, features = [], []

    t0 = time.time()
    for i, pac_start in enumerate(coords):
        for _ in range(per_start):
            pellet_positions = _random_pellets(base_spec, pellet_density, pellet_count)
            if pellet_count is not None:
                pellet_positions = pellet_positions[:pellet_count]
            spec = replace(
                base_spec,
                pacman_start=pac_start,
                pellet_positions=pellet_positions,
                random_seed=random.randint(0, 1_000_000_000),
            )
            feats = [
                w * h,
                len(spec.pellet_positions) / (w * h),
                np.mean([x for x, _ in spec.pellet_positions]) if spec.pellet_positions else 0,
                np.mean([y for _, y in spec.pellet_positions]) if spec.pellet_positions else 0,
                np.var([x for x, _ in spec.pellet_positions]) if spec.pellet_positions else 0,
                np.var([y for _, y in spec.pellet_positions]) if spec.pellet_positions else 0,
                *pac_start,
            ]
            maze_specs.append(spec)
            features.append(feats)

        if (i + 1) % 10 == 0 or i == n_positions - 1:
            print(f"  progress: {i+1}/{n_positions} starts processed")

    print(f"Total generated samples: {len(maze_specs):,} in {time.time() - t0:.1f}s")
    print("Clustering lattice samples...")

    if len(maze_specs) < n_clusters * 2:
        print(f"⚠️ Too few samples ({len(maze_specs)}) for {n_clusters} clusters — skipping clustering")
        clusters = {0: maze_specs}
    else:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10_000, n_init="auto", max_iter=100)
        start = time.time()
        labels = kmeans.fit_predict(features)
        clusters = {i: [] for i in range(n_clusters)}
        for spec, label in zip(maze_specs, labels):
            clusters[label].append(spec)
        clusters = {k: v for k, v in clusters.items() if v}
        print(f"Total clusters with samples: {len(clusters)}")
        print(f"Clustering done in {time.time() - start:.1f}s")

    # --- visualize a few random cluster samples ---
    # print(f"Previewing {min(100, len(clusters))} cluster samples before continuing:")
    # for i, (key, specs) in enumerate(clusters.items()):
    #     if i >= 10:
    #         break
    #     print(f"\nCluster {key} ({len(specs)} samples)")
    #     for _ in range(5):
    #         sample_spec = random.choice(specs)
    #         ascii_output = _render_ascii(sample_spec)
    #         print(ascii_output)
    #     input("Press Enter to view next cluster...")

    # print("✅ Cluster preview complete.")

    cluster_keys = list(clusters.keys())
    idx = 0

    def sampler():
        """Cycle through clusters, sampling evenly from each cluster."""
        nonlocal idx
        if not cluster_keys:
            return random.choice(maze_specs)
        key = cluster_keys[idx % len(cluster_keys)]
        idx += 1
        return random.choice(clusters[key])

    return sampler
