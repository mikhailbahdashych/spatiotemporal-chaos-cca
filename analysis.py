"""
Analysis tools for chaos, bifurcation, and synchronization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from cellular_automaton import ContinuousCellularAutomaton
from scipy import signal
from scipy.stats import entropy


def compute_lyapunov_exponent(
    ca: ContinuousCellularAutomaton,
    steps: int = 1000,
    transient: int = 200,
    perturbation: float = 1e-8
) -> float:
    """
    Compute largest Lyapunov exponent for the CA system.

    The Lyapunov exponent measures sensitivity to initial conditions.
    Positive values indicate chaos.

    Parameters:
    -----------
    ca : ContinuousCellularAutomaton
        Cellular automaton to analyze
    steps : int
        Number of steps to compute
    transient : int
        Initial steps to skip (let system stabilize)
    perturbation : float
        Initial perturbation magnitude

    Returns:
    --------
    float : Lyapunov exponent
    """
    # Create initial state
    initial_state = np.random.rand(*ca.grid_size)

    # Create perturbed state
    perturbed_state = initial_state + perturbation * np.random.randn(*ca.grid_size)
    perturbed_state = np.clip(perturbed_state, 0, 1)

    # Create two CA instances
    ca1 = ContinuousCellularAutomaton(
        grid_size=ca.grid_size,
        r=ca.r,
        epsilon=ca.epsilon,
        boundary=ca.boundary,
        topology=ca.topology,
        random_neighbors=ca.random_neighbors,
        n_neighbors=ca.n_neighbors
    )
    ca1.reset(initial_state)

    ca2 = ContinuousCellularAutomaton(
        grid_size=ca.grid_size,
        r=ca.r,
        epsilon=ca.epsilon,
        boundary=ca.boundary,
        topology=ca.topology,
        random_neighbors=ca.random_neighbors,
        n_neighbors=ca.n_neighbors
    )
    ca2.reset(perturbed_state)

    # Skip transient
    ca1.run(transient, record_history=False)
    ca2.run(transient, record_history=False)

    # Compute Lyapunov exponent
    lyapunov_sum = 0.0

    for _ in range(steps):
        ca1.step()
        ca2.step()

        # Compute distance
        distance = np.sqrt(np.mean((ca1.grid - ca2.grid) ** 2))

        if distance > 0:
            lyapunov_sum += np.log(distance / perturbation)

            # Renormalize to prevent overflow
            ca2.grid = ca1.grid + perturbation * (ca2.grid - ca1.grid) / distance
            ca2.grid = np.clip(ca2.grid, 0, 1)

    return lyapunov_sum / steps


def bifurcation_diagram(
    r_values: np.ndarray,
    grid_size: Tuple[int, int] = (100, 100),
    epsilon: float = 0.5,
    transient: int = 200,
    sample_steps: int = 100,
    n_sample_cells: int = 100,
    **ca_kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate bifurcation diagram by varying parameter r.

    Parameters:
    -----------
    r_values : np.ndarray
        Array of r values to test
    grid_size : tuple
        Size of the grid
    epsilon : float
        Coupling strength
    transient : int
        Steps to skip before sampling
    sample_steps : int
        Number of steps to sample
    n_sample_cells : int
        Number of random cells to sample
    **ca_kwargs : dict
        Additional arguments for CA

    Returns:
    --------
    r_array, values_array : tuple of arrays
        Arrays for plotting bifurcation diagram
    """
    r_list = []
    values_list = []

    for r in r_values:
        print(f"Computing r = {r:.3f}...", end='\r')

        ca = ContinuousCellularAutomaton(
            grid_size=grid_size,
            r=r,
            epsilon=epsilon,
            **ca_kwargs
        )

        # Skip transient
        ca.run(transient, record_history=False)

        # Sample values
        for _ in range(sample_steps):
            ca.step()

            # Sample random cells
            for _ in range(n_sample_cells):
                i = np.random.randint(0, grid_size[0])
                j = np.random.randint(0, grid_size[1])
                r_list.append(r)
                values_list.append(ca.grid[i, j])

    print("\nBifurcation diagram complete!")

    return np.array(r_list), np.array(values_list)


def plot_bifurcation_diagram(
    r_values: np.ndarray,
    values: np.ndarray,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Bifurcation Diagram"
):
    """
    Plot bifurcation diagram.

    Parameters:
    -----------
    r_values : np.ndarray
        Parameter values
    values : np.ndarray
        Corresponding state values
    figsize : tuple
        Figure size
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(r_values, values, ',k', alpha=0.3, markersize=0.5)
    ax.set_xlabel('Growth Parameter r', fontsize=12)
    ax.set_ylabel('Cell Density', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    return fig


def compute_synchronization_metrics(ca: ContinuousCellularAutomaton) -> dict:
    """
    Compute various synchronization metrics.

    Parameters:
    -----------
    ca : ContinuousCellularAutomaton
        Cellular automaton with history

    Returns:
    --------
    dict : Dictionary with synchronization metrics
    """
    grid = ca.grid

    # Spatial variance (low = high synchronization)
    spatial_variance = np.var(grid)

    # Order parameter: average deviation from mean
    mean_val = np.mean(grid)
    order_parameter = np.mean(np.abs(grid - mean_val))

    # Coefficient of variation
    cv = np.std(grid) / (np.mean(grid) + 1e-10)

    # Entropy (high = low synchronization)
    hist, _ = np.histogram(grid.flatten(), bins=50, range=(0, 1), density=True)
    hist = hist + 1e-10  # Avoid log(0)
    sync_entropy = entropy(hist)

    return {
        'spatial_variance': spatial_variance,
        'order_parameter': order_parameter,
        'coefficient_variation': cv,
        'entropy': sync_entropy,
        'synchronization_index': 1.0 - order_parameter  # Higher = more sync
    }


def synchronization_analysis(
    epsilon_values: np.ndarray,
    r: float = 3.8,
    grid_size: Tuple[int, int] = (100, 100),
    steps: int = 500,
    transient: int = 200,
    **ca_kwargs
) -> dict:
    """
    Analyze synchronization as function of coupling strength epsilon.

    Parameters:
    -----------
    epsilon_values : np.ndarray
        Array of epsilon values to test
    r : float
        Growth parameter
    grid_size : tuple
        Size of the grid
    steps : int
        Total simulation steps
    transient : int
        Transient steps to skip
    **ca_kwargs : dict
        Additional CA arguments

    Returns:
    --------
    dict : Dictionary with arrays of metrics vs epsilon
    """
    results = {
        'epsilon': [],
        'spatial_variance': [],
        'order_parameter': [],
        'coefficient_variation': [],
        'entropy': [],
        'synchronization_index': []
    }

    for eps in epsilon_values:
        print(f"Computing ε = {eps:.3f}...", end='\r')

        ca = ContinuousCellularAutomaton(
            grid_size=grid_size,
            r=r,
            epsilon=eps,
            **ca_kwargs
        )

        # Run simulation
        ca.run(steps, record_history=False)

        # Compute metrics (average over last few steps)
        metrics_list = []
        for _ in range(20):  # Average over 20 steps
            ca.step()
            metrics = compute_synchronization_metrics(ca)
            metrics_list.append(metrics)

        # Average metrics
        avg_metrics = {
            key: np.mean([m[key] for m in metrics_list])
            for key in metrics_list[0].keys()
        }

        results['epsilon'].append(eps)
        for key, value in avg_metrics.items():
            results[key].append(value)

    print("\nSynchronization analysis complete!")

    return {key: np.array(val) for key, val in results.items()}


def plot_synchronization_analysis(
    results: dict,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Plot synchronization analysis results.

    Parameters:
    -----------
    results : dict
        Results from synchronization_analysis
    figsize : tuple
        Figure size
    """
    epsilon = results['epsilon']

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Spatial variance
    axes[0].plot(epsilon, results['spatial_variance'], 'o-', linewidth=2)
    axes[0].set_xlabel('Coupling Strength ε')
    axes[0].set_ylabel('Spatial Variance')
    axes[0].set_title('Spatial Variance vs ε\n(Lower = More Synchronized)')
    axes[0].grid(True, alpha=0.3)

    # Order parameter
    axes[1].plot(epsilon, results['order_parameter'], 'o-', linewidth=2, color='orange')
    axes[1].set_xlabel('Coupling Strength ε')
    axes[1].set_ylabel('Order Parameter')
    axes[1].set_title('Order Parameter vs ε\n(Lower = More Synchronized)')
    axes[1].grid(True, alpha=0.3)

    # Synchronization index
    axes[2].plot(epsilon, results['synchronization_index'], 'o-', linewidth=2, color='green')
    axes[2].set_xlabel('Coupling Strength ε')
    axes[2].set_ylabel('Synchronization Index')
    axes[2].set_title('Synchronization Index vs ε\n(Higher = More Synchronized)')
    axes[2].grid(True, alpha=0.3)

    # Entropy
    axes[3].plot(epsilon, results['entropy'], 'o-', linewidth=2, color='red')
    axes[3].set_xlabel('Coupling Strength ε')
    axes[3].set_ylabel('Entropy')
    axes[3].set_title('Entropy vs ε\n(Lower = More Synchronized)')
    axes[3].grid(True, alpha=0.3)

    return fig


def lyapunov_spectrum(
    r_values: np.ndarray,
    epsilon: float = 0.5,
    grid_size: Tuple[int, int] = (50, 50),
    steps: int = 500,
    transient: int = 200,
    **ca_kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Lyapunov exponent spectrum over range of r values.

    Parameters:
    -----------
    r_values : np.ndarray
        Array of r values to test
    epsilon : float
        Coupling strength
    grid_size : tuple
        Size of the grid
    steps : int
        Number of steps for computation
    transient : int
        Transient steps
    **ca_kwargs : dict
        Additional CA arguments

    Returns:
    --------
    r_values, lyapunov_values : tuple of arrays
    """
    lyapunov_values = []

    for r in r_values:
        print(f"Computing Lyapunov for r = {r:.3f}...", end='\r')

        ca = ContinuousCellularAutomaton(
            grid_size=grid_size,
            r=r,
            epsilon=epsilon,
            **ca_kwargs
        )

        lyap = compute_lyapunov_exponent(ca, steps=steps, transient=transient)
        lyapunov_values.append(lyap)

    print("\nLyapunov spectrum complete!")

    return r_values, np.array(lyapunov_values)


def plot_lyapunov_spectrum(
    r_values: np.ndarray,
    lyapunov_values: np.ndarray,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot Lyapunov exponent spectrum.

    Parameters:
    -----------
    r_values : np.ndarray
        Parameter values
    lyapunov_values : np.ndarray
        Lyapunov exponents
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(r_values, lyapunov_values, 'o-', linewidth=2)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Chaos threshold')
    ax.fill_between(r_values, 0, lyapunov_values,
                     where=(lyapunov_values > 0),
                     alpha=0.3, color='red', label='Chaotic')
    ax.fill_between(r_values, lyapunov_values, 0,
                     where=(lyapunov_values < 0),
                     alpha=0.3, color='blue', label='Stable')

    ax.set_xlabel('Growth Parameter r', fontsize=12)
    ax.set_ylabel('Lyapunov Exponent', fontsize=12)
    ax.set_title('Lyapunov Exponent Spectrum\n(Positive = Chaos)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def power_spectrum_analysis(
    ca: ContinuousCellularAutomaton,
    i: Optional[int] = None,
    j: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Compute and plot power spectrum of cell trajectory.

    Parameters:
    -----------
    ca : ContinuousCellularAutomaton
        Cellular automaton with history
    i, j : int
        Cell coordinates (if None, use center)
    figsize : tuple
        Figure size
    """
    if i is None or j is None:
        i, j = ca.grid_size[0] // 2, ca.grid_size[1] // 2

    trajectory = ca.get_cell_trajectory(i, j)

    # Compute power spectrum
    freqs, psd = signal.periodogram(trajectory)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Time series
    ax1.plot(trajectory)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Cell Density')
    ax1.set_title(f'Time Series - Cell ({i}, {j})')
    ax1.grid(True, alpha=0.3)

    # Power spectrum
    ax2.semilogy(freqs, psd)
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_title('Power Spectrum')
    ax2.grid(True, alpha=0.3)

    return fig


def compare_topologies(
    topologies: List[dict],
    r: float = 3.8,
    epsilon: float = 0.5,
    grid_size: Tuple[int, int] = (100, 100),
    steps: int = 500,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Compare different network topologies.

    Parameters:
    -----------
    topologies : list of dict
        Each dict contains topology configuration
        Example: {'name': 'Moore', 'topology': 'moore', 'random_neighbors': False}
    r : float
        Growth parameter
    epsilon : float
        Coupling strength
    grid_size : tuple
        Size of the grid
    steps : int
        Number of simulation steps
    figsize : tuple
        Figure size
    """
    n_topo = len(topologies)
    fig, axes = plt.subplots(2, n_topo, figsize=figsize)

    if n_topo == 1:
        axes = axes.reshape(-1, 1)

    for idx, topo_config in enumerate(topologies):
        name = topo_config.pop('name')
        print(f"Running {name} topology...")

        ca = ContinuousCellularAutomaton(
            grid_size=grid_size,
            r=r,
            epsilon=epsilon,
            **topo_config
        )

        ca.run(steps, record_history=True)

        # Plot final grid state
        im = axes[0, idx].imshow(ca.grid, cmap='viridis', vmin=0, vmax=1)
        axes[0, idx].set_title(f'{name}\nFinal State')
        plt.colorbar(im, ax=axes[0, idx])

        # Plot mean time series
        mean_series = ca.get_time_series('mean')
        axes[1, idx].plot(mean_series, linewidth=1)
        axes[1, idx].set_xlabel('Time Step')
        axes[1, idx].set_ylabel('Mean Density')
        axes[1, idx].set_title(f'{name}\nMean Evolution')
        axes[1, idx].grid(True, alpha=0.3)

    return fig
