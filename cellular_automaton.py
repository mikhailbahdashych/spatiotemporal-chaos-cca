"""
Continuous Cellular Automaton Engine
Implements Coupled Map Lattice (CML) for studying spatiotemporal chaos
"""

import numpy as np
from numba import jit
from typing import Tuple, Optional, Callable


class ContinuousCellularAutomaton:
    """
    Continuous Cellular Automaton using logistic map with spatial coupling.

    Cell update rule: x_i(t+1) = (1-ε) * f(x_i(t)) + (ε/N) * Σf(x_neighbors(t))
    Where f(x) = r * x * (1-x) is the logistic map
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (100, 100),
        r: float = 3.8,
        epsilon: float = 0.5,
        boundary: str = 'periodic',
        topology: str = 'moore',
        random_neighbors: bool = False,
        n_neighbors: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the cellular automaton.

        Parameters:
        -----------
        grid_size : tuple
            (height, width) of the grid
        r : float
            Growth parameter for logistic map (typically 2.5-4.0)
        epsilon : float
            Coupling strength (0=isolated, 1=fully coupled)
        boundary : str
            'periodic', 'fixed', or 'reflective'
        topology : str
            'moore' (8 neighbors), 'vonneumann' (4 neighbors), or 'random'
        random_neighbors : bool
            If True, randomly select neighbors for each cell
        n_neighbors : int
            Number of neighbors (for random topology)
        seed : int
            Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.r = r
        self.epsilon = epsilon
        self.boundary = boundary
        self.topology = topology
        self.random_neighbors = random_neighbors
        self.n_neighbors = n_neighbors

        if seed is not None:
            np.random.seed(seed)

        # Initialize grid with random values [0, 1]
        self.grid = np.random.rand(*grid_size)
        self.history = [self.grid.copy()]

        # Build neighbor map
        self._build_neighbor_map()

    def _build_neighbor_map(self):
        """Build mapping of neighbors for each cell."""
        h, w = self.grid_size

        if self.random_neighbors:
            # Random neighbor selection
            if self.n_neighbors is None:
                self.n_neighbors = 8
            self.neighbor_map = self._build_random_neighbor_map()
        else:
            # Regular topology
            if self.topology == 'moore':
                # 8 neighbors
                self.offsets = [(-1, -1), (-1, 0), (-1, 1),
                               (0, -1),           (0, 1),
                               (1, -1),  (1, 0),  (1, 1)]
            elif self.topology == 'vonneumann':
                # 4 neighbors
                self.offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
            else:
                raise ValueError(f"Unknown topology: {self.topology}")

    def _build_random_neighbor_map(self) -> np.ndarray:
        """Build random neighbor connections."""
        h, w = self.grid_size
        neighbor_map = np.zeros((h, w, self.n_neighbors, 2), dtype=np.int32)

        for i in range(h):
            for j in range(w):
                # Random positions
                neighbors_i = np.random.randint(0, h, self.n_neighbors)
                neighbors_j = np.random.randint(0, w, self.n_neighbors)
                neighbor_map[i, j, :, 0] = neighbors_i
                neighbor_map[i, j, :, 1] = neighbors_j

        return neighbor_map

    def logistic_map(self, x: np.ndarray) -> np.ndarray:
        """Apply logistic map: f(x) = r * x * (1-x)"""
        return self.r * x * (1.0 - x)

    def step(self):
        """Perform one simulation step."""
        new_grid = np.zeros_like(self.grid)
        h, w = self.grid_size

        if self.random_neighbors:
            # Use random neighbor map
            for i in range(h):
                for j in range(w):
                    # Local contribution
                    local = self.logistic_map(self.grid[i, j])

                    # Neighbor contribution
                    neighbor_sum = 0.0
                    for k in range(self.n_neighbors):
                        ni, nj = self.neighbor_map[i, j, k]
                        neighbor_sum += self.logistic_map(self.grid[ni, nj])

                    neighbor_avg = neighbor_sum / self.n_neighbors

                    # Coupled map lattice update
                    new_grid[i, j] = (1 - self.epsilon) * local + self.epsilon * neighbor_avg
        else:
            # Regular topology
            for i in range(h):
                for j in range(w):
                    # Local contribution
                    local = self.logistic_map(self.grid[i, j])

                    # Neighbor contribution
                    neighbor_sum = 0.0
                    n_valid = 0

                    for di, dj in self.offsets:
                        ni, nj = i + di, j + dj

                        # Handle boundaries
                        if self.boundary == 'periodic':
                            ni = ni % h
                            nj = nj % w
                        elif self.boundary == 'fixed':
                            if ni < 0 or ni >= h or nj < 0 or nj >= w:
                                continue

                        neighbor_sum += self.logistic_map(self.grid[ni, nj])
                        n_valid += 1

                    if n_valid > 0:
                        neighbor_avg = neighbor_sum / n_valid
                        # Coupled map lattice update
                        new_grid[i, j] = (1 - self.epsilon) * local + self.epsilon * neighbor_avg
                    else:
                        new_grid[i, j] = local

        # Clip values to [0, 1]
        new_grid = np.clip(new_grid, 0.0, 1.0)

        self.grid = new_grid
        self.history.append(self.grid.copy())

    def run(self, steps: int, record_history: bool = True):
        """
        Run simulation for given number of steps.

        Parameters:
        -----------
        steps : int
            Number of time steps to simulate
        record_history : bool
            If True, store entire history (memory intensive)
        """
        if not record_history:
            self.history = []

        for _ in range(steps):
            self.step()
            if not record_history:
                # Only keep last state
                self.history = [self.grid.copy()]

    def reset(self, initial_state: Optional[np.ndarray] = None):
        """Reset the automaton to initial state."""
        if initial_state is not None:
            self.grid = initial_state.copy()
        else:
            self.grid = np.random.rand(*self.grid_size)
        self.history = [self.grid.copy()]

    def get_population_stats(self) -> dict:
        """Calculate statistical measures of current population."""
        return {
            'mean': np.mean(self.grid),
            'std': np.std(self.grid),
            'var': np.var(self.grid),
            'min': np.min(self.grid),
            'max': np.max(self.grid),
            'entropy': -np.sum(self.grid * np.log(self.grid + 1e-10))
        }

    def get_history_array(self) -> np.ndarray:
        """Get history as 3D array (time, height, width)."""
        return np.array(self.history)

    def get_time_series(self, metric: str = 'mean') -> np.ndarray:
        """
        Get time series of global metric.

        Parameters:
        -----------
        metric : str
            'mean', 'std', 'var', 'min', 'max', or 'entropy'
        """
        time_series = []
        for state in self.history:
            if metric == 'mean':
                time_series.append(np.mean(state))
            elif metric == 'std':
                time_series.append(np.std(state))
            elif metric == 'var':
                time_series.append(np.var(state))
            elif metric == 'min':
                time_series.append(np.min(state))
            elif metric == 'max':
                time_series.append(np.max(state))
            elif metric == 'entropy':
                time_series.append(-np.sum(state * np.log(state + 1e-10)))

        return np.array(time_series)

    def get_cell_trajectory(self, i: int, j: int) -> np.ndarray:
        """Get time series for a specific cell."""
        return np.array([state[i, j] for state in self.history])
