from typing import Protocol

import numpy as np


class Problem(Protocol):
    def __call__(self, new_cost: float, best_cost: float) -> bool:
        pass
    

class Minimize(Problem):
    def __call__(self, new_cost: float, best_cost: float) -> bool:
        return new_cost < best_cost


class Maximize(Problem):
    def __call__(self, new_cost: float, best_cost: float) -> bool:
        return new_cost > best_cost


class Solution:
    def __init__(self, n_companies: int, n_projects: int):
        self.n_companies = n_companies
        self.n_projects = n_projects
        self.x = np.random.choice(range(1, n_projects + 1), size=n_companies, replace=False)

    def move(self):
        i, j = np.random.choice(self.n_companies, size=2, replace=False)
        self.x[i], self.x[j] = self.x[j], self.x[i]

    def sparse(self) -> np.ndarray:
        sparse_solution = np.zeros((self.n_companies, self.n_projects), dtype=int)
        sparse_solution[np.arange(self.n_companies), self.x.flatten() - 1] = 1
        return sparse_solution

    def __str__(self):
        return str(self.x)
    
    def __repr__(self):
        return str(self.x)


class ObjectiveFunction:
    def __init__(self, cost_matrix: np.ndarray):
        self.cost_matrix = cost_matrix

    def __call__(self, solution: Solution) -> float:
        sparse_solution = solution.sparse()
        solution_matrix = sparse_solution * self.cost_matrix
        return solution_matrix.sum()
