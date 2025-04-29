from typing import Callable

import numpy as np

from utils import Problem, Solution

class HillClimbing:
    def __init__(self, problem: Problem, objective_function: Callable, cost_matrix: np.ndarray, max_iterations: int = 1000, patience: int = 100):
        self.problem = problem
        self.objective_function = objective_function
        self.cost_matrix = cost_matrix
        self.max_iterations = max_iterations
        self.patience = patience
        self.n_iterations = 0
        self._called = False
        
        self.initial_solution = self.best_solution = Solution(cost_matrix.shape[0], cost_matrix.shape[1])
        self.best_cost = self.objective_function(self.best_solution)
        self.history = {
            "best_solution": [],
            "best_cost": [],
        }

    def run(self):

        if self._called:
            raise RuntimeError("This method can only be called once.")

        current_solution = self.initial_solution
        current_patience = 0

        while self.n_iterations < self.max_iterations or current_patience < self.patience:

            current_solution.move()
            new_cost = self.objective_function(self.best_solution)

            if self.problem(new_cost, self.best_cost):
                self.best_solution = current_solution
                self.best_cost = new_cost

            self.history["best_solution"].append(self.best_solution)
            self.history["best_cost"].append(self.best_cost)
                
            self.n_iterations += 1
            current_patience += 1
        
        self._called = True

    def __str__(self):
        return f"Iteration: {self.n_iterations}\n\nBest cost: {self.best_cost}\n\nBest solution: {self.best_solution.x}\n\nAllocation:\n{self.best_solution.sparse()}"
    
    def __repr__(self):
        return f"Iteration: {self.n_iterations}\n\nBest cost: {self.best_cost}\n\nBest solution: {self.best_solution.x}\n\nAllocation:\n{self.best_solution.sparse()}"
    