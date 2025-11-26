"""
Main entry point for Continuous Cellular Automaton experiments.

For full analysis, see experiments.py (can be converted to Jupyter notebook).
"""

from cellular_automaton import ContinuousCellularAutomaton
from visualization import create_summary_plot
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("Continuous Cellular Automaton - Spatiotemporal Chaos")
    print("=" * 60)
    print()
    print("Running a quick demonstration...")
    print()

    # Create a simple CA
    ca = ContinuousCellularAutomaton(
        grid_size=(100, 100),
        r=3.8,
        epsilon=0.5,
        boundary='periodic',
        topology='moore',
        seed=42
    )

    print(f"Grid size: {ca.grid_size}")
    print(f"Growth parameter r: {ca.r}")
    print(f"Coupling strength ε: {ca.epsilon}")
    print(f"Topology: {ca.topology}")
    print()

    # Run simulation
    print("Running simulation for 300 steps...")
    ca.run(steps=300, record_history=True)
    print("Simulation complete!")
    print()

    # Display statistics
    stats = ca.get_population_stats()
    print("Final population statistics:")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Std:  {stats['std']:.4f}")
    print(f"  Min:  {stats['min']:.4f}")
    print(f"  Max:  {stats['max']:.4f}")
    print()

    # Create summary plot
    print("Generating summary plot...")
    fig = create_summary_plot(ca)
    plt.show()
    print()
    print("=" * 60)
    print("For full analysis, run: python experiments.py")
    print("Or convert to notebook: jupytext --to notebook experiments.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
