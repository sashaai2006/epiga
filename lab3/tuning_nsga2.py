from dataclasses import dataclass
from math import ceil, sqrt
from time import perf_counter
from typing import Literal

import numpy as np


N_FEATURES = 10
N_CLASSES = 3
N_SAMPLES = 800
DATA_SEED = 202610
TRAIN_FRACTION = 0.7


@dataclass(frozen=True)
class Gene:
    name: str
    low: float
    high: float
    kind: Literal["int", "float"]


GENES = (
    Gene("hidden", 0, 16, "int"),
    Gene("log10_lr", -3.0, -1.0, "float"),
    Gene("epochs", 5, 150, "int"),
    Gene("l2", 0.0, 0.05, "float"),
    Gene("log2_batch", 5, 7, "int"),
)
GENOME_SIZE = len(GENES)
LOWER = np.array([g.low for g in GENES])
UPPER = np.array([g.high for g in GENES])
INT_MASK = np.array([g.kind == "int" for g in GENES])


@dataclass(frozen=True)
class ClassificationData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    data_seed: int = DATA_SEED

    @property
    def n_features(self) -> int:
        return self.x_train.shape[1]

    @property
    def n_classes(self) -> int:
        return int(self.y_train.max()) + 1


def generate_data(seed: int = DATA_SEED, n_samples: int = N_SAMPLES,
                  n_features: int = N_FEATURES, n_classes: int = N_CLASSES) -> ClassificationData:
    rng = np.random.default_rng(seed)
    x = np.empty((n_samples, n_features))
    y = np.empty(n_samples, dtype=int)
    per_class = n_samples // n_classes
    for c in range(n_classes):
        center = rng.uniform(-1.5, 1.5, size=n_features)
        block = slice(c * per_class, (c + 1) * per_class if c < n_classes - 1 else n_samples)
        x[block] = center + rng.normal(0.0, 1.2, size=(n_samples - c * per_class if c == n_classes - 1 else per_class, n_features))
        y[block] = c
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    x = (x - mean) / std
    order = rng.permutation(n_samples)
    x, y = x[order], y[order]
    n_train = int(n_samples * TRAIN_FRACTION)
    return ClassificationData(x_train=x[:n_train], y_train=y[:n_train], x_test=x[n_train:], y_test=y[n_train:])


def decode(genome: np.ndarray) -> dict:
    genome = clip_genome(genome)
    return {
        "hidden": int(round(genome[0])),
        "lr": float(10.0 ** genome[1]),
        "epochs": int(round(genome[2])),
        "l2": float(genome[3]),
        "batch_size": int(2 ** round(genome[4])),
    }


def clip_genome(genome: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(genome, dtype=float), LOWER, UPPER)
    clipped[INT_MASK] = np.round(clipped[INT_MASK])
    return clipped


def n_params(hidden: int, n_features: int, n_classes: int) -> int:
    if hidden == 0:
        return (n_features + 1) * n_classes
    return (n_features + 1) * hidden + (hidden + 1) * n_classes


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_and_eval(params: dict, data: ClassificationData) -> tuple[float, float, int, float]:
    """Обучает модель и возвращает (ошибка на тесте, стоимость обучения, число параметров, время обучения, с)."""
    started = perf_counter()
    hidden = params["hidden"]
    n_features, n_classes = data.n_features, data.n_classes
    count = n_params(hidden, n_features, n_classes)
    rng = np.random.default_rng(data.data_seed)
    y_onehot = np.eye(n_classes)[data.y_train]
    if hidden == 0:
        w = 0.01 * rng.standard_normal((n_features, n_classes))
        b = np.zeros(n_classes)
    else:
        w1 = 0.01 * rng.standard_normal((n_features, hidden))
        b1 = np.zeros(hidden)
        w2 = 0.01 * rng.standard_normal((hidden, n_classes))
        b2 = np.zeros(n_classes)
    n_train = len(data.x_train)
    batch = params["batch_size"]
    n_batches = ceil(n_train / batch)
    for _ in range(params["epochs"]):
        order = rng.permutation(n_train)
        for i in range(n_batches):
            idx = order[i * batch:(i + 1) * batch]
            xb, yb = data.x_train[idx], y_onehot[idx]
            m = len(idx)
            if hidden == 0:
                p = _softmax(xb @ w + b)
                dz = (p - yb) / m
                w -= params["lr"] * (xb.T @ dz + params["l2"] * w)
                b -= params["lr"] * dz.sum(axis=0)
            else:
                h = np.tanh(xb @ w1 + b1)
                p = _softmax(h @ w2 + b2)
                dz = (p - yb) / m
                w2 -= params["lr"] * (h.T @ dz + params["l2"] * w2)
                b2 -= params["lr"] * dz.sum(axis=0)
                dh = dz @ w2.T * (1.0 - h * h)
                w1 -= params["lr"] * (xb.T @ dh + params["l2"] * w1)
                b1 -= params["lr"] * dh.sum(axis=0)
    if hidden == 0:
        predictions = np.argmax(_softmax(data.x_test @ w + b), axis=1)
    else:
        predictions = np.argmax(_softmax(np.tanh(data.x_test @ w1 + b1) @ w2 + b2), axis=1)
    test_error = float(np.mean(predictions != data.y_test))
    cost = float(params["epochs"] * n_batches * count)
    return test_error, cost, count, perf_counter() - started


def evaluate(genome: np.ndarray, data: ClassificationData) -> np.ndarray:
    return np.array(train_and_eval(decode(genome), data)[:3])


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(objectives: np.ndarray) -> list[np.ndarray]:
    n = len(objectives)
    if n == 0:
        return []
    le = (objectives[:, None, :] <= objectives[None, :, :]).all(axis=2)
    lt = (objectives[:, None, :] < objectives[None, :, :]).any(axis=2)
    dominates_matrix = le & lt & ~np.eye(n, dtype=bool)
    dominated_count = dominates_matrix.sum(axis=0)
    fronts: list[np.ndarray] = []
    current = np.flatnonzero(dominated_count == 0)
    remaining = np.ones(n, dtype=bool)
    while len(current):
        fronts.append(current)
        remaining[current] = False
        dominated_count -= dominates_matrix[current].sum(axis=0)
        candidates = np.flatnonzero(remaining & (dominated_count == 0))
        current = candidates
    return fronts


def crowding_distance(front_objectives: np.ndarray) -> np.ndarray:
    n, m = front_objectives.shape
    if n <= 2:
        return np.full(n, np.inf)
    distance = np.zeros(n)
    for k in range(m):
        order = np.argsort(front_objectives[:, k])
        values = front_objectives[order, k]
        span = values[-1] - values[0]
        distance[order[0]] = distance[order[-1]] = np.inf
        if span > 0:
            distance[order[1:-1]] += (values[2:] - values[:-2]) / span
    return distance


def hypervolume_2d(points: np.ndarray, ref: tuple[float, float]) -> float:
    points = points[(points[:, 0] <= ref[0]) & (points[:, 1] <= ref[1])]
    if len(points) == 0:
        return 0.0
    order = np.argsort(points[:, 0])
    x = points[order, 0]
    y = np.minimum.accumulate(points[order, 1])
    right = np.minimum(np.append(x[1:], ref[0]), ref[0])
    widths = np.maximum(right - x, 0.0)
    heights = np.maximum(ref[1] - y, 0.0)
    return float(np.sum(widths * heights))


def hypervolume_3d(points: np.ndarray, ref: tuple[float, float, float]) -> float:
    points = points[np.all(points <= np.array(ref), axis=1)]
    if len(points) == 0:
        return 0.0
    order = np.argsort(points[:, 2])
    z = points[order, 2]
    volume = 0.0
    running = []
    for i in range(len(z)):
        upper = min(z[i + 1], ref[2]) if i + 1 < len(z) else ref[2]
        running.append(points[order[i]])
        volume += (upper - z[i]) * hypervolume_2d(np.array(running), (ref[0], ref[1]))
    return float(volume)


def reference_point(data: ClassificationData) -> tuple[float, float, float]:
    max_params = n_params(int(UPPER[0]), data.n_features, data.n_classes)
    max_batches = ceil(len(data.x_train) / 2 ** int(LOWER[4]))
    max_cost = UPPER[2] * max_batches * max_params
    return 1.01, 1.1 * max_cost, 1.1 * max_params


def sbx_crossover(parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator,
                  crossover_prob: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() >= crossover_prob:
        return parent1.copy(), parent2.copy()
    child1, child2 = parent1.copy(), parent2.copy()
    for j in range(GENOME_SIZE):
        if rng.random() < 0.5 and abs(parent1[j] - parent2[j]) > 1e-12:
            y1, y2 = sorted((parent1[j], parent2[j]))
            beta = 1.0 + 2.0 * (y1 - LOWER[j]) / (y2 - y1)
            alpha = 2.0 - beta ** -(eta + 1.0)
            u = rng.random()
            beta1 = (u * alpha) ** (1.0 / (eta + 1.0)) if u <= 1.0 / alpha else (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta + 1.0))
            beta = 1.0 + 2.0 * (UPPER[j] - y2) / (y2 - y1)
            alpha = 2.0 - beta ** -(eta + 1.0)
            u = rng.random()
            beta2 = (u * alpha) ** (1.0 / (eta + 1.0)) if u <= 1.0 / alpha else (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta + 1.0))
            child1[j] = 0.5 * ((y1 + y2) - beta1 * (y2 - y1))
            child2[j] = 0.5 * ((y1 + y2) + beta2 * (y2 - y1))
    return clip_genome(child1), clip_genome(child2)


def polynomial_mutation(genome: np.ndarray, rng: np.random.Generator,
                        mutation_prob: float, eta: float) -> np.ndarray:
    child = genome.copy()
    for j in range(GENOME_SIZE):
        if rng.random() < mutation_prob:
            u = rng.random()
            delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0 if u < 0.5 else 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
            span = UPPER[j] - LOWER[j]
            child[j] = child[j] + delta * span
    return clip_genome(child)


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 60
    budget: int = 1000
    crossover_prob: float = 0.9
    mutation_prob: float = 0.15
    sbx_eta: float = 15.0
    mutation_eta: float = 20.0
    seed: int = 42
    method: Literal["nsga2", "weighted_sum"] = "nsga2"
    weights: tuple[float, float, float] = (0.6, 0.2, 0.2)

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size должен быть не меньше 2")
        if self.budget < self.population_size:
            raise ValueError("budget должен быть не меньше population_size")
        if not 0 <= self.crossover_prob <= 1 or not 0 <= self.mutation_prob <= 1:
            raise ValueError("Вероятности операторов должны быть в [0, 1]")


class Archive:
    """Все оценённые решения: генотипы, цели и кэш вычислений."""

    def __init__(self, data: ClassificationData):
        self.data = data
        self.genomes: list[np.ndarray] = []
        self.objectives: list[np.ndarray] = []
        self.cache: dict[tuple, np.ndarray] = {}
        self.eval_seconds: list[float] = []
        self.n_evals = 0

    def evaluate(self, genome: np.ndarray) -> np.ndarray:
        key = tuple(clip_genome(genome))
        self.n_evals += 1
        if key not in self.cache:
            started = perf_counter()
            self.cache[key] = evaluate(np.array(key), self.data)
            self.eval_seconds.append(perf_counter() - started)
            self.genomes.append(np.array(key))
            self.objectives.append(self.cache[key])
        return self.cache[key]

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.genomes), np.vstack(self.objectives)


def tournament(rng: np.random.Generator, rank: np.ndarray, crowding: np.ndarray) -> int:
    candidates = rng.integers(0, len(rank), size=2)
    better = candidates[0] if (rank[candidates[0]], -crowding[candidates[0]]) <= \
        (rank[candidates[1]], -crowding[candidates[1]]) else candidates[1]
    return int(better)


def run_nsga2(data: ClassificationData, config: GAConfig) -> dict:
    started = perf_counter()
    rng = np.random.default_rng(config.seed)
    archive = Archive(data)
    ref = reference_point(data)
    population = [LOWER + rng.random(GENOME_SIZE) * (UPPER - LOWER) for _ in range(config.population_size)]
    population = [clip_genome(p) for p in population]
    objectives = np.vstack([archive.evaluate(p) for p in population])
    history_evals: list[int] = []
    history_hv: list[float] = []

    def record() -> None:
        _, all_objectives = archive.as_arrays()
        fronts = non_dominated_sort(all_objectives)
        history_evals.append(archive.n_evals)
        history_hv.append(hypervolume_3d(all_objectives[fronts[0]], ref))

    record()
    while archive.n_evals < config.budget:
        size = min(config.population_size, config.budget - archive.n_evals)
        fronts = non_dominated_sort(objectives)
        rank = np.empty(len(population), dtype=int)
        crowding = np.empty(len(population))
        for r, front in enumerate(fronts):
            rank[front] = r
            crowding[front] = crowding_distance(objectives[front])
        offspring: list[np.ndarray] = []
        while len(offspring) < size:
            p1 = population[tournament(rng, rank, crowding)]
            p2 = population[tournament(rng, rank, crowding)]
            c1, c2 = sbx_crossover(p1, p2, rng, config.crossover_prob, config.sbx_eta)
            offspring.append(polynomial_mutation(c1, rng, config.mutation_prob, config.mutation_eta))
            if len(offspring) < size:
                offspring.append(polynomial_mutation(c2, rng, config.mutation_prob, config.mutation_eta))
        offspring_objectives = np.vstack([archive.evaluate(c) for c in offspring])
        combined = population + offspring
        combined_objectives = np.vstack([objectives, offspring_objectives])
        fronts = non_dominated_sort(combined_objectives)
        next_indices: list[int] = []
        for front in fronts:
            if len(next_indices) + len(front) <= config.population_size:
                next_indices.extend(front.tolist())
            else:
                distances = crowding_distance(combined_objectives[front])
                order = np.argsort(-distances)
                next_indices.extend(front[order[: config.population_size - len(next_indices)]].tolist())
                break
        population = [combined[i] for i in next_indices]
        objectives = combined_objectives[next_indices]
        record()

    genomes, all_objectives = archive.as_arrays()
    fronts = non_dominated_sort(all_objectives)
    front = fronts[0]
    return {
        "method": config.method,
        "seed": config.seed,
        "n_evals": archive.n_evals,
        "elapsed_seconds": perf_counter() - started,
        "mean_eval_seconds": float(np.mean(archive.eval_seconds)) if archive.eval_seconds else 0.0,
        "hypervolume": history_hv[-1],
        "front_size": len(front),
        "front_genomes": genomes[front].tolist(),
        "front_objectives": all_objectives[front].tolist(),
        "history_evals": history_evals,
        "history_hv": history_hv,
    }


def run_weighted_sum(data: ClassificationData, config: GAConfig) -> dict:
    """Однокритериальный ГА над взвешенной суммой; фронт строится из архива всех оценённых решений."""
    started = perf_counter()
    rng = np.random.default_rng(config.seed)
    archive = Archive(data)
    ref = np.array(reference_point(data))
    weights = np.array(config.weights)

    def scalar(values: np.ndarray) -> float:
        return float(weights @ (values / ref))

    population = [clip_genome(LOWER + rng.random(GENOME_SIZE) * (UPPER - LOWER))
                  for _ in range(config.population_size)]
    objectives = np.vstack([archive.evaluate(p) for p in population])
    values = np.array([scalar(o) for o in objectives])
    best = float(values.min())
    history_evals: list[int] = [archive.n_evals]
    history_hv: list[float] = []

    def archive_hv() -> float:
        _, all_objectives = archive.as_arrays()
        fronts = non_dominated_sort(all_objectives)
        return hypervolume_3d(all_objectives[fronts[0]], tuple(ref))

    history_hv.append(archive_hv())
    while archive.n_evals < config.budget:
        size = min(config.population_size, config.budget - archive.n_evals)
        elite_count = min(1, size)
        if size <= elite_count:
            size = elite_count + 1
        elite_indices = np.argsort(values)[:elite_count]
        next_population = [population[int(i)].copy() for i in elite_indices]
        while len(next_population) < size:
            candidates = rng.integers(0, len(population), size=2)
            p1 = population[int(candidates[np.argmin(values[candidates])])]
            candidates = rng.integers(0, len(population), size=2)
            p2 = population[int(candidates[np.argmin(values[candidates])])]
            c1, c2 = sbx_crossover(p1, p2, rng, config.crossover_prob, config.sbx_eta)
            next_population.append(polynomial_mutation(c1, rng, config.mutation_prob, config.mutation_eta))
            if len(next_population) < size:
                next_population.append(polynomial_mutation(c2, rng, config.mutation_prob, config.mutation_eta))
        new_count = size - len(elite_indices)
        if new_count > 0:
            new_objectives = np.vstack([archive.evaluate(c) for c in next_population[len(elite_indices):]])
            objectives = np.vstack([objectives[elite_indices], new_objectives])
        else:
            objectives = objectives[elite_indices]
        values = np.array([scalar(o) for o in objectives])
        population = next_population
        best = min(best, float(values.min()))
        history_evals.append(archive.n_evals)
        history_hv.append(archive_hv())

    genomes, all_objectives = archive.as_arrays()
    fronts = non_dominated_sort(all_objectives)
    front = fronts[0]
    return {
        "method": config.method,
        "seed": config.seed,
        "n_evals": archive.n_evals,
        "elapsed_seconds": perf_counter() - started,
        "mean_eval_seconds": float(np.mean(archive.eval_seconds)) if archive.eval_seconds else 0.0,
        "best_scalar": best,
        "hypervolume": history_hv[-1],
        "front_size": len(front),
        "front_genomes": genomes[front].tolist(),
        "front_objectives": all_objectives[front].tolist(),
        "history_evals": history_evals,
        "history_hv": history_hv,
    }


def run_random_search(data: ClassificationData, config: GAConfig) -> dict:
    started = perf_counter()
    rng = np.random.default_rng(config.seed)
    archive = Archive(data)
    ref = reference_point(data)
    history_evals: list[int] = []
    history_hv: list[float] = []
    while archive.n_evals < config.budget:
        size = min(config.population_size, config.budget - archive.n_evals)
        for _ in range(size):
            archive.evaluate(LOWER + rng.random(GENOME_SIZE) * (UPPER - LOWER))
        genomes, all_objectives = archive.as_arrays()
        fronts = non_dominated_sort(all_objectives)
        history_evals.append(archive.n_evals)
        history_hv.append(hypervolume_3d(all_objectives[fronts[0]], ref))
    genomes, all_objectives = archive.as_arrays()
    fronts = non_dominated_sort(all_objectives)
    front = fronts[0]
    return {
        "method": "random_search",
        "seed": config.seed,
        "n_evals": archive.n_evals,
        "elapsed_seconds": perf_counter() - started,
        "mean_eval_seconds": float(np.mean(archive.eval_seconds)) if archive.eval_seconds else 0.0,
        "hypervolume": history_hv[-1],
        "front_size": len(front),
        "front_genomes": genomes[front].tolist(),
        "front_objectives": all_objectives[front].tolist(),
        "history_evals": history_evals,
        "history_hv": history_hv,
    }


def run_method(data: ClassificationData, config: GAConfig) -> dict:
    if config.method == "nsga2":
        return run_nsga2(data, config)
    if config.method == "weighted_sum":
        return run_weighted_sum(data, config)
    raise ValueError(f"Неизвестный метод: {config.method}")
