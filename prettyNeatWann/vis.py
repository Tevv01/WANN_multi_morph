"""
plot_wann_results.py

Unified plotting script for WANN experiments.

Supports:
- Single-run plots (fitness, complexity)
- Multi-run plots (median + IQR)
- Single-network visualization

Uses ONLY existing visualization utilities:
- neatVis.py
- lplot.py
- viewInd.py
"""

import os
import matplotlib.pyplot as plt

from vis.neatVis import viewFitFile, viewReps
from vis.viewInd import viewInd

# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# SINGLE-RUN PLOTS
# ------------------------------------------------------------

def plot_single_run(stats_file, val, out_name, title):
    """
    Plot a metric from ONE stats file.
    Uses viewFitFile internally.
    """
    fig, ax = viewFitFile(stats_file, val=val)

    ax.set_xlabel("Evaluations")
    ax.set_ylabel(val)
    ax.set_title(title)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, out_name))
    plt.close(fig)


def plot_single_run_fitness(stats_file, out_name="single_fitness.pdf"):
    plot_single_run(
        stats_file=stats_file,
        val="Fit",
        out_name=out_name,
        title="Training Performance (Single Run)"
    )


def plot_single_run_complexity(stats_file, out_name="single_complexity.pdf"):
    plot_single_run(
        stats_file=stats_file,
        val="Conn",
        out_name=out_name,
        title="Network Complexity (Single Run)"
    )

# ------------------------------------------------------------
# MULTI-RUN PLOTS (requires multiple stats files)
# ------------------------------------------------------------

def plot_multi_run(prefixes, labels, val, out_name, title):
    """
    Plot median + IQR across multiple runs.
    Uses viewReps internally.
    """
    fig, ax = viewReps(
        prefix=prefixes,
        label=labels,
        val=val,
        title=title
    )

    ax.set_xlabel("Evaluations")
    ax.set_ylabel(val)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, out_name))
    plt.close(fig)


def plot_multi_run_fitness(prefixes, labels, out_name="multi_fitness.pdf"):
    plot_multi_run(
        prefixes=prefixes,
        labels=labels,
        val="Fit",
        out_name=out_name,
        title="Training Performance (Multiple Runs)"
    )


def plot_multi_run_complexity(prefixes, labels, out_name="multi_complexity.pdf"):
    plot_multi_run(
        prefixes=prefixes,
        labels=labels,
        val="Conn",
        out_name=out_name,
        title="Network Complexity (Multiple Runs)"
    )

# ------------------------------------------------------------
# SINGLE NETWORK VISUALIZATION
# ------------------------------------------------------------

def plot_single_network(best_file, out_name="best_network.pdf",
                        task_name="evogym_walker"):
    """
    Visualize ONE network from a .out file.
    Uses viewInd internally.
    """
    fig, ax = viewInd(best_file, taskName=task_name)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, out_name))
    plt.close(fig)

# ------------------------------------------------------------
# EXAMPLE USAGE
# ------------------------------------------------------------

if __name__ == "__main__":

    # -------- SINGLE RUN --------
    plot_single_run_fitness(
        stats_file="log/baseline_20_01_50_200_5_stats.out",
        out_name="baseline_single_fitness.pdf"
    )

    plot_single_run_complexity(
        stats_file="log/baseline_20_01_50_200_5_stats.out",
        out_name="baseline_single_complexity.pdf"
    )

    plot_single_network(
        best_file="log/baseline_20_01_50_200_5_best.out",
        out_name="baseline_best_network.pdf"
    )

    # -------- MULTI RUN (only use when you actually have multiple runs) --------
    # plot_multi_run_fitness(
    #     prefixes=[
    #         "log/baseline_run1",
    #         "log/baseline_run2",
    #         "log/baseline_run3"
    #     ],
    #     labels=["Baseline"],
    #     out_name="baseline_multi_fitness.pdf"
    # )

    print("All plots saved to:", OUTPUT_DIR)
