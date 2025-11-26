"""
Visualization tools for cellular automaton analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, List, Tuple
from cellular_automaton import ContinuousCellularAutomaton


def plot_grid_state(
    grid: np.ndarray,
    title: str = "Grid State",
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (8, 8),
    colorbar: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot a single grid state as heatmap.

    Parameters:
    -----------
    grid : np.ndarray
        2D grid to plot
    title : str
        Plot title
    cmap : str
        Colormap name
    figsize : tuple
        Figure size
    colorbar : bool
        Show colorbar
    ax : plt.Axes
        Existing axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    if colorbar:
        plt.colorbar(im, ax=ax, label='Cell Density')

    return ax


def plot_grid_evolution(
    ca: ContinuousCellularAutomaton,
    timesteps: Optional[List[int]] = None,
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Plot grid states at different timesteps.

    Parameters:
    -----------
    ca : ContinuousCellularAutomaton
        Cellular automaton object with history
    timesteps : list
        Specific timesteps to plot (if None, plot evenly spaced)
    cmap : str
        Colormap name
    figsize : tuple
        Figure size
    """
    history = ca.get_history_array()
    n_steps = len(history)

    if timesteps is None:
        # Plot 6 evenly spaced timesteps
        timesteps = np.linspace(0, n_steps - 1, 6, dtype=int)

    n_plots = len(timesteps)
    cols = 3
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n_plots > 1 else [axes]

    for idx, t in enumerate(timesteps):
        if idx < len(axes):
            plot_grid_state(
                history[t],
                title=f"t = {t}",
                cmap=cmap,
                colorbar=False,
                ax=axes[idx]
            )

    # Remove extra subplots
    for idx in range(n_plots, len(axes)):
        fig.delaxes(axes[idx])

    # Add single colorbar
    fig.colorbar(
        axes[0].images[0],
        ax=axes[:n_plots],
        label='Cell Density',
        fraction=0.046,
        pad=0.04
    )

    return fig


def animate_ca(
    ca: ContinuousCellularAutomaton,
    interval: int = 50,
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
):
    """
    Create animation of cellular automaton evolution.

    Parameters:
    -----------
    ca : ContinuousCellularAutomaton
        Cellular automaton with history
    interval : int
        Milliseconds between frames
    cmap : str
        Colormap name
    figsize : tuple
        Figure size
    save_path : str
        If provided, save animation to file
    """
    history = ca.get_history_array()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(history[0], cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Cell Density')
    title = ax.set_title(f"t = 0, r={ca.r:.2f}, ε={ca.epsilon:.2f}")

    def update(frame):
        im.set_array(history[frame])
        title.set_text(f"t = {frame}, r={ca.r:.2f}, ε={ca.epsilon:.2f}")
        return [im, title]

    anim = FuncAnimation(
        fig, update, frames=len(history),
        interval=interval, blit=True, repeat=True
    )

    if save_path:
        anim.save(save_path, writer='pillow')

    return anim


def plot_time_series(
    ca: ContinuousCellularAutomaton,
    metrics: List[str] = ['mean', 'std'],
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot time series of global metrics.

    Parameters:
    -----------
    ca : ContinuousCellularAutomaton
        Cellular automaton with history
    metrics : list
        List of metrics to plot ('mean', 'std', 'var', etc.)
    figsize : tuple
        Figure size
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)

    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        time_series = ca.get_time_series(metric)
        axes[idx].plot(time_series, linewidth=1)
        axes[idx].set_ylabel(metric.capitalize())
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time Step')
    axes[0].set_title(f'Global Metrics (r={ca.r:.2f}, ε={ca.epsilon:.2f})')
    return fig


def plot_cell_trajectories(
    ca: ContinuousCellularAutomaton,
    n_cells: int = 5,
    figsize: Tuple[int, int] = (12, 6),
    random_cells: bool = True
):
    """
    Plot trajectories of individual cells over time.

    Parameters:
    -----------
    ca : ContinuousCellularAutomaton
        Cellular automaton with history
    n_cells : int
        Number of cells to plot
    figsize : tuple
        Figure size
    random_cells : bool
        If True, select random cells; otherwise use evenly spaced
    """
    fig, ax = plt.subplots(figsize=figsize)

    h, w = ca.grid_size

    if random_cells:
        cells = [(np.random.randint(0, h), np.random.randint(0, w))
                 for _ in range(n_cells)]
    else:
        step = max(h, w) // (n_cells + 1)
        cells = [(i * step, i * step) for i in range(1, n_cells + 1)]

    for i, j in cells:
        trajectory = ca.get_cell_trajectory(i, j)
        ax.plot(trajectory, label=f'Cell ({i}, {j})', alpha=0.7)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cell Density')
    ax.set_title(f'Individual Cell Trajectories (r={ca.r:.2f}, ε={ca.epsilon:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_phase_space(
    ca: ContinuousCellularAutomaton,
    i: int = None,
    j: int = None,
    figsize: Tuple[int, int] = (8, 8)
):
    """
    Plot phase space diagram x(t) vs x(t+1).

    Parameters:
    -----------
    ca : ContinuousCellularAutomaton
        Cellular automaton with history
    i, j : int
        Cell coordinates (if None, use center cell)
    figsize : tuple
        Figure size
    """
    if i is None or j is None:
        i, j = ca.grid_size[0] // 2, ca.grid_size[1] // 2

    trajectory = ca.get_cell_trajectory(i, j)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(trajectory[:-1], trajectory[1:], 'o-', alpha=0.5, markersize=2)
    ax.set_xlabel('x(t)')
    ax.set_ylabel('x(t+1)')
    ax.set_title(f'Phase Space - Cell ({i}, {j}), r={ca.r:.2f}, ε={ca.epsilon:.2f}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add diagonal line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='x(t)=x(t+1)')
    ax.legend()

    return fig


def plot_spatial_correlation(
    grid: np.ndarray,
    max_distance: int = 50,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot spatial correlation function.

    Parameters:
    -----------
    grid : np.ndarray
        2D grid state
    max_distance : int
        Maximum distance to compute correlation
    figsize : tuple
        Figure size
    """
    h, w = grid.shape
    correlations = []
    distances = range(1, min(max_distance, min(h, w) // 2))

    center = grid - np.mean(grid)
    var = np.var(grid)

    for d in distances:
        # Horizontal correlation
        corr_h = np.mean(center[:, :-d] * center[:, d:]) / var
        # Vertical correlation
        corr_v = np.mean(center[:-d, :] * center[d:, :]) / var
        # Average
        correlations.append((corr_h + corr_v) / 2)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(distances, correlations, 'o-')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Correlation')
    ax.set_title('Spatial Correlation Function')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    return fig


def create_summary_plot(
    ca: ContinuousCellularAutomaton,
    figsize: Tuple[int, int] = (16, 10)
):
    """
    Create comprehensive summary plot with multiple views.

    Parameters:
    -----------
    ca : ContinuousCellularAutomaton
        Cellular automaton with history
    figsize : tuple
        Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Current grid state
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    im = ax1.imshow(ca.grid, cmap='viridis', vmin=0, vmax=1)
    ax1.set_title(f'Current State (t={len(ca.history)-1})')
    plt.colorbar(im, ax=ax1, label='Density')

    # Time series - mean
    ax2 = fig.add_subplot(gs[0, 2])
    mean_series = ca.get_time_series('mean')
    ax2.plot(mean_series, linewidth=1)
    ax2.set_ylabel('Mean Density')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Population Mean')

    # Time series - std
    ax3 = fig.add_subplot(gs[1, 2])
    std_series = ca.get_time_series('std')
    ax3.plot(std_series, linewidth=1, color='orange')
    ax3.set_ylabel('Std Dev')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Population Std Dev')

    # Phase space for center cell
    ax4 = fig.add_subplot(gs[2, 0])
    i, j = ca.grid_size[0] // 2, ca.grid_size[1] // 2
    trajectory = ca.get_cell_trajectory(i, j)
    ax4.plot(trajectory[:-1], trajectory[1:], 'o-', alpha=0.5, markersize=2)
    ax4.set_xlabel('x(t)')
    ax4.set_ylabel('x(t+1)')
    ax4.set_title(f'Phase Space Cell ({i},{j})')
    ax4.grid(True, alpha=0.3)

    # Cell trajectories
    ax5 = fig.add_subplot(gs[2, 1:3])
    for k in range(5):
        i = np.random.randint(0, ca.grid_size[0])
        j = np.random.randint(0, ca.grid_size[1])
        traj = ca.get_cell_trajectory(i, j)
        ax5.plot(traj, alpha=0.6, linewidth=1)
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Cell Density')
    ax5.set_title('Sample Cell Trajectories')
    ax5.grid(True, alpha=0.3)

    fig.suptitle(f'CA Summary: r={ca.r:.2f}, ε={ca.epsilon:.2f}, '
                 f'Grid={ca.grid_size[0]}x{ca.grid_size[1]}',
                 fontsize=14, fontweight='bold')

    return fig
