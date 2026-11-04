from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd


DAYS = ("Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье")
SHIFTS = ("Утро", "День", "Ночь")


@dataclass(frozen=True)
class DutyProblem:
    employees: tuple[str, ...]
    preferences: np.ndarray
    max_shifts: int = 4
    data_seed: int = 202610

    def __post_init__(self) -> None:
        if len(self.employees) < 5:
            raise ValueError("Нужно не менее 5 сотрудников")
        if self.preferences.shape != (len(self.employees), 21):
            raise ValueError("Матрица предпочтений должна иметь форму (n_employees, 21)")
        if not np.isin(self.preferences, (0, 1, 4, 9)).all():
            raise ValueError("Допустимые значения предпочтений: 0, 1, 4, 9")
        if self.max_shifts < 3:
            raise ValueError("Лимит смен должен быть не меньше 3")

    @property
    def n_employees(self) -> int:
        return len(self.employees)

    @property
    def n_slots(self) -> int:
        return self.preferences.shape[1]


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 60
    budget: int = 6_000
    tournament_size: int = 3
    crossover_prob: float = 0.9
    mutation_prob: float = 0.15
    mutation: Literal["reassign", "bit_flip"] = "reassign"
    constraint_handler: Literal["repair", "penalty"] = "repair"
    penalty_weight: float = 1_000.0
    elitism: int = 2
    seed: int = 42

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size должен быть не меньше 2")
        if self.budget < self.population_size:
            raise ValueError("budget должен быть не меньше population_size")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("Некорректный размер турнира")
        if not 0 <= self.elitism < self.population_size:
            raise ValueError("elitism должен быть в [0, population_size)")
        if not 0 <= self.crossover_prob <= 1 or not 0 <= self.mutation_prob <= 1:
            raise ValueError("Вероятности операторов должны быть в [0, 1]")
        if self.penalty_weight <= 0:
            raise ValueError("penalty_weight должен быть положительным")


def generate_problem(seed: int = 202610, n_employees: int = 7) -> DutyProblem:
    rng = np.random.default_rng(seed)
    employees = tuple(f"Сотрудник {i + 1}" for i in range(n_employees))
    preferences = rng.choice((0, 1, 4), size=(n_employees, 21), p=(0.25, 0.55, 0.20))
    unavailable = rng.random(preferences.shape) < 0.10
    preferences[unavailable] = 9
    problem = DutyProblem(employees=employees, preferences=preferences, data_seed=seed)
    rng = np.random.default_rng(seed + 1)
    constructive_schedule(problem, rng)
    return problem


def problem_to_dict(problem: DutyProblem) -> dict:
    return {
        "employees": list(problem.employees),
        "days": list(DAYS),
        "shifts": list(SHIFTS),
        "max_shifts": problem.max_shifts,
        "data_seed": problem.data_seed,
        "preference_codes": {"0": "желательно", "1": "нейтрально", "4": "нежелательно", "9": "недоступен"},
        "preferences": problem.preferences.tolist(),
    }


def validate_genome(problem: DutyProblem, genome: np.ndarray) -> np.ndarray:
    genome = np.asarray(genome)
    if genome.shape != (problem.n_employees, problem.n_slots):
        raise ValueError("Некорректная форма генотипа")
    if not np.isin(genome, (0, 1)).all():
        raise ValueError("Генотип должен быть бинарным")
    return genome.astype(np.int8, copy=False)


def solution_metrics(problem: DutyProblem, genome: np.ndarray) -> dict:
    x = validate_genome(problem, genome).astype(int)
    by_day = x.reshape(problem.n_employees, 7, 3)
    loads = x.sum(axis=1)
    coverage = int(np.abs(x.sum(axis=0) - 1).sum())
    daily = int(np.maximum(by_day.sum(axis=2) - 1, 0).sum())
    rest = int(sum(x[:, day * 3 + 2] @ x[:, (day + 1) * 3] for day in range(6)))
    limit = int(np.maximum(loads - problem.max_shifts, 0).sum())
    unavailable = int((x * (problem.preferences == 9)).sum())
    preference = int((x * problem.preferences).sum())
    imbalance = float(2 * np.square(loads - 3).sum())
    hard = coverage + daily + rest + limit + unavailable
    return {
        "coverage": coverage,
        "daily": daily,
        "rest": rest,
        "limit": limit,
        "unavailable": unavailable,
        "hard_violations": hard,
        "preference": preference,
        "imbalance": imbalance,
        "objective": float(preference + imbalance),
        "loads": loads.tolist(),
    }


def is_feasible(problem: DutyProblem, genome: np.ndarray) -> bool:
    return solution_metrics(problem, genome)["hard_violations"] == 0


def fitness(problem: DutyProblem, genome: np.ndarray, config: GAConfig) -> float:
    metrics = solution_metrics(problem, genome)
    if config.constraint_handler == "penalty":
        return metrics["objective"] + config.penalty_weight * metrics["hard_violations"]
    return metrics["objective"]


def random_genome(problem: DutyProblem, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros((problem.n_employees, problem.n_slots), dtype=np.int8)
    x[rng.integers(problem.n_employees, size=problem.n_slots), np.arange(problem.n_slots)] = 1
    return x


def uniform_shift_crossover(
    parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    mask = rng.random(parent1.shape[1]) < 0.5
    child1, child2 = parent1.copy(), parent2.copy()
    child1[:, mask] = parent2[:, mask]
    child2[:, mask] = parent1[:, mask]
    return child1, child2


def mutate(problem: DutyProblem, genome: np.ndarray, config: GAConfig, rng: np.random.Generator) -> np.ndarray:
    child = genome.copy()
    for slot in np.flatnonzero(rng.random(problem.n_slots) < config.mutation_prob):
        if config.mutation == "bit_flip":
            employee = int(rng.integers(problem.n_employees))
            child[employee, slot] = 1 - child[employee, slot]
        else:
            current = int(np.argmax(child[:, slot]))
            employee = int(rng.integers(problem.n_employees - 1))
            if employee >= current:
                employee += 1
            child[:, slot] = 0
            child[employee, slot] = 1
    return child


def constructive_schedule(
    problem: DutyProblem,
    rng: np.random.Generator,
    preferred: np.ndarray | None = None,
    randomized: bool = False,
    max_attempts: int = 100,
) -> np.ndarray:
    preferred = None if preferred is None else validate_genome(problem, preferred)
    for attempt in range(max_attempts):
        x = np.zeros((problem.n_employees, problem.n_slots), dtype=np.int8)
        loads = np.zeros(problem.n_employees, dtype=int)
        failed = False
        for slot in range(problem.n_slots):
            day, shift = divmod(slot, 3)
            candidates = []
            for employee in range(problem.n_employees):
                if problem.preferences[employee, slot] == 9 or loads[employee] >= problem.max_shifts:
                    continue
                if x[employee, day * 3 : (day + 1) * 3].any():
                    continue
                if shift == 0 and day > 0 and x[employee, (day - 1) * 3 + 2]:
                    continue
                balance_delta = 2 * ((loads[employee] + 1 - 3) ** 2 - (loads[employee] - 3) ** 2)
                score = float(problem.preferences[employee, slot] + balance_delta)
                if preferred is not None:
                    score += 0.0 if preferred[employee, slot] else 1.5
                if randomized:
                    score += rng.random() * (0.5 + 0.15 * attempt)
                else:
                    score += rng.random() * 1e-6
                candidates.append((score, employee))
            if not candidates:
                failed = True
                break
            employee = min(candidates)[1]
            x[employee, slot] = 1
            loads[employee] += 1
        if not failed and is_feasible(problem, x):
            return x
    raise RuntimeError("Конструктивный алгоритм не нашёл допустимое расписание")


def repair(problem: DutyProblem, genome: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return constructive_schedule(problem, rng, preferred=genome, randomized=True)


def schedule_table(problem: DutyProblem, genome: np.ndarray) -> pd.DataFrame:
    x = validate_genome(problem, genome)
    table = []
    for day in range(7):
        row = {}
        for shift in range(3):
            assigned = np.flatnonzero(x[:, day * 3 + shift])
            row[SHIFTS[shift]] = ", ".join(problem.employees[i] for i in assigned) if len(assigned) else "—"
        table.append(row)
    return pd.DataFrame(table, index=DAYS)


class GeneticAlgorithm:
    def __init__(self, problem: DutyProblem, config: GAConfig):
        self.problem = problem
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.n_evals = 0
        self.best_fitness = float("inf")
        self.best_x: np.ndarray | None = None
        self.best_history: list[float] = []
        self.eval_history: list[int] = []

    def _prepare(self, genome: np.ndarray) -> np.ndarray:
        if self.config.constraint_handler == "repair":
            return repair(self.problem, genome, self.rng)
        return genome

    def _evaluate(self, population: list[np.ndarray]) -> np.ndarray:
        values = np.array([fitness(self.problem, x, self.config) for x in population])
        self.n_evals += len(population)
        index = int(np.argmin(values))
        if values[index] < self.best_fitness:
            self.best_fitness = float(values[index])
            self.best_x = population[index].copy()
        self.best_history.append(self.best_fitness)
        self.eval_history.append(self.n_evals)
        return values

    def _select(self, population: list[np.ndarray], values: np.ndarray) -> np.ndarray:
        indices = self.rng.choice(len(population), size=min(self.config.tournament_size, len(population)), replace=False)
        return population[int(indices[np.argmin(values[indices])])].copy()

    def run(self) -> dict:
        started = perf_counter()
        cfg = self.config
        population = [self._prepare(random_genome(self.problem, self.rng)) for _ in range(cfg.population_size)]
        values = self._evaluate(population)
        while self.n_evals < cfg.budget:
            size = min(cfg.population_size, cfg.budget - self.n_evals)
            elite_count = min(cfg.elitism, size)
            elite_indices = np.argsort(values)[:elite_count]
            next_population = [population[int(i)].copy() for i in elite_indices]
            while len(next_population) < size:
                parent1 = self._select(population, values)
                parent2 = self._select(population, values)
                if self.rng.random() < cfg.crossover_prob:
                    child1, child2 = uniform_shift_crossover(parent1, parent2, self.rng)
                else:
                    child1, child2 = parent1, parent2
                next_population.append(self._prepare(mutate(self.problem, child1, cfg, self.rng)))
                if len(next_population) < size:
                    next_population.append(self._prepare(mutate(self.problem, child2, cfg, self.rng)))
            population = next_population
            values = self._evaluate(population)
        if self.best_x is None:
            raise RuntimeError("Алгоритм завершился без оценки решений")
        metrics = solution_metrics(self.problem, self.best_x)
        return {
            "best_x": self.best_x.tolist(),
            "best_fitness": self.best_fitness,
            "best_objective": metrics["objective"],
            "feasible": metrics["hard_violations"] == 0,
            "hard_violations": metrics["hard_violations"],
            "preference": metrics["preference"],
            "imbalance": metrics["imbalance"],
            "loads": metrics["loads"],
            "n_evals": self.n_evals,
            "elapsed_seconds": perf_counter() - started,
            "best_history": self.best_history,
            "eval_history": self.eval_history,
        }


def random_feasible_search(problem: DutyProblem, config: GAConfig) -> dict:
    started = perf_counter()
    rng = np.random.default_rng(config.seed)
    best_fitness = float("inf")
    best_x = None
    best_history = []
    eval_history = []
    n_evals = 0
    while n_evals < config.budget:
        size = min(config.population_size, config.budget - n_evals)
        for _ in range(size):
            candidate = repair(problem, random_genome(problem, rng), rng)
            value = solution_metrics(problem, candidate)["objective"]
            n_evals += 1
            if value < best_fitness:
                best_fitness = value
                best_x = candidate.copy()
        best_history.append(float(best_fitness))
        eval_history.append(n_evals)
    if best_x is None:
        raise RuntimeError("Случайный поиск завершился без оценки решений")
    metrics = solution_metrics(problem, best_x)
    return {
        "best_x": best_x.tolist(),
        "best_fitness": float(best_fitness),
        "best_objective": metrics["objective"],
        "feasible": True,
        "hard_violations": 0,
        "preference": metrics["preference"],
        "imbalance": metrics["imbalance"],
        "loads": metrics["loads"],
        "n_evals": n_evals,
        "elapsed_seconds": perf_counter() - started,
        "best_history": best_history,
        "eval_history": eval_history,
    }


def save_instance(problem: DutyProblem, path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(problem_to_dict(problem), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
