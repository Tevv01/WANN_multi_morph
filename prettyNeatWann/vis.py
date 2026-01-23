import pandas as pd

STAT_COLS = [
    "x_scale",
    "fit_med",
    "fit_max",
    "fit_top",
    "fit_peak",
    "node_med",
    "conn_med"
]

def load_stats(path):
    return pd.read_csv(
        path,
        header=None,
        names=STAT_COLS
    )

def load_objvals(path):
    return pd.read_csv(
        path,
        header=None,
        names=["fitness", "peak_fitness", "connections"]
    )
