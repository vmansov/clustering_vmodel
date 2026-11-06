# ============================================================
# ENERGY RELAXATION FIGURE
# Rows: Random / Cluster
# Columns: selected initial disorder D values
#
# Main axes:
#   mean tissue energy E(t) for every mutant fraction
#
# Inset:
#   energy relaxation rate dE/dt
#
# IMPORTANT:
# dE/dt is NOT literally the mechanical force.
# In overdamped gradient dynamics, dE/dt -> 0 is consistent with
# vanishing motion / force balance, but if you want to plot the
# actual force you should save ||grad E|| (or the force norm)
# directly from the simulation.
# ============================================================

import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyprojroot import here
from vertex_lite.custom import load_data_v3 as load_data


# ============================================================
# USER SETTINGS
# ============================================================

PCTS = [
    0.0,
    0.025, 0.05, 0.075, 0.10, 0.125,
    0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30
]

# Choose the D values you want to show.
NOISES_TO_PLOT = [0.1, 0.5]

# Final time shown on the x axis.
# Use None to show all available times.
TIME_MAX = 15000

# Optional starting time.
TIME_MIN = 0

# Smooth ONLY the curve used to estimate dE/dt in the inset.
# Set to 1 for no smoothing.
# Must be an odd integer >= 1.
SMOOTH_WINDOW = 11

# How much of the inset y-range is determined robustly.
# Values outside these percentiles are clipped only visually;
# the data themselves are not modified.
INSET_PERCENTILES = (1, 99)

# If True, all panels in a row use the same y limits for energy.
SHARE_ENERGY_Y_WITHIN_ARRANGEMENT = True

RANDOM_COLOR = "#709E0D"
CLUSTER_COLOR = "#A53379"

ARRANGEMENT_COLORS = {
    "Random": RANDOM_COLOR,
    "Cluster": CLUSTER_COLOR,
}

SAVE_DIR = here("results/analysis/energy_relaxation")
os.makedirs(SAVE_DIR, exist_ok=True)

OUTPUT_BASENAME = "energy_plateau_with_relaxation_insets"


# ============================================================
# STYLE
# ============================================================

sns.set_theme(
    style="white",
    context="paper",
    rc={
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    },
)


def make_fraction_cmap(base_color):
    """Light-to-dark cmap for mutant fraction."""
    return sns.blend_palette(
        [
            sns.light_palette(base_color, n_colors=8)[2],
            base_color,
        ],
        as_cmap=True,
    )


CMAPS = {
    "Random": make_fraction_cmap(RANDOM_COLOR),
    "Cluster": make_fraction_cmap(CLUSTER_COLOR),
}

PCT_NORM = mpl.colors.Normalize(
    vmin=min(PCTS),
    vmax=max(PCTS),
)


# ============================================================
# DATA LOADING
# ============================================================

def load_energy_data(arrangement):
    """
    arrangement:
        'Random'  -> results/resultados_random
        'Cluster' -> results/resultados_cluster
    """

    if arrangement == "Cluster":
        rpath = here("results/resultados_cluster")
    elif arrangement == "Random":
        rpath = here("results/resultados_random")
    else:
        raise ValueError("arrangement must be 'Random' or 'Cluster'")

    dfs = []

    for pct in PCTS:
        pattern = (
            f"{rpath}/"
            f"noise_*/"
            f"pct_{pct:g}/"
            f"sim_*/"
            f"interface.parquet"
        )

        d = load_data(
            pattern,
            columns=[
                "time",
                "energy",
            ],
        )

        if d is not None and not d.empty:
            d = d.copy()
            d["arrangement"] = arrangement
            dfs.append(d)

    if len(dfs) == 0:
        raise ValueError(f"No energy data found for {arrangement}")

    df = pd.concat(dfs, ignore_index=True)

    # Remove duplicated saved times if they exist.
    group_cols = [
        "arrangement",
        "mesh_type",
        "pct_mutant",
        "sim_id",
        "time",
    ]

    df = (
        df
        .groupby(group_cols, as_index=False)
        .agg(energy=("energy", "mean"))
        .sort_values(group_cols)
        .reset_index(drop=True)
    )

    return df


# ============================================================
# TIME FILTER
# ============================================================

def restrict_time(df):
    out = df.copy()

    if TIME_MIN is not None:
        out = out[out["time"] >= TIME_MIN]

    if TIME_MAX is not None:
        out = out[out["time"] <= TIME_MAX]

    return out.copy()


# ============================================================
# SMOOTHING + ENERGY DERIVATIVE
# ============================================================

def smooth_series(values, window):
    """
    Centered rolling mean used only before differentiating.
    This reduces numerical noise in dE/dt without modifying
    the energy curves in the main panel.
    """
    values = pd.Series(np.asarray(values, dtype=float))

    if window is None or window <= 1:
        return values.to_numpy()

    window = int(window)

    if window % 2 == 0:
        window += 1

    return (
        values
        .rolling(
            window=window,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy()
    )


def calculate_mean_energy_and_relaxation(df):
    """
    1) Average energy across stochastic realisations at each time.
    2) Smooth the MEAN curve only for derivative estimation.
    3) Compute dE/dt using the actual time coordinates.

    The main energy plot always uses the unsmoothed mean energy.
    """

    mean_energy = (
        df
        .groupby(
            [
                "arrangement",
                "mesh_type",
                "pct_mutant",
                "time",
            ],
            as_index=False,
        )
        .agg(
            energy_mean=("energy", "mean"),
            energy_sem=("energy", "sem"),
            n_sim=("sim_id", "nunique"),
        )
        .sort_values(
            [
                "arrangement",
                "mesh_type",
                "pct_mutant",
                "time",
            ]
        )
        .reset_index(drop=True)
    )

    pieces = []

    group_cols = [
        "arrangement",
        "mesh_type",
        "pct_mutant",
    ]

    for keys, g in mean_energy.groupby(group_cols):
        g = g.sort_values("time").copy()

        t = g["time"].to_numpy(dtype=float)
        E = g["energy_mean"].to_numpy(dtype=float)

        E_for_derivative = smooth_series(E, SMOOTH_WINDOW)

        if len(g) >= 2 and np.unique(t).size >= 2:
            dE_dt = np.gradient(E_for_derivative, t)
        else:
            dE_dt = np.full(len(g), np.nan)

        g["energy_for_derivative"] = E_for_derivative
        g["dE_dt"] = dE_dt

        pieces.append(g)

    return pd.concat(pieces, ignore_index=True)


# ============================================================
# ROBUST INSET LIMITS
# ============================================================

def robust_symmetric_ylim(values, percentiles=(1, 99)):
    """
    Symmetric y-limits around zero using robust percentiles,
    so one early numerical spike does not flatten the whole inset.
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return (-1, 1)

    lo, hi = np.percentile(values, percentiles)
    span = max(abs(lo), abs(hi))

    if span <= 0:
        span = max(np.max(np.abs(values)), 1e-12)

    return (-1.10 * span, 1.10 * span)


# ============================================================
# PLOT
# ============================================================

def plot_energy_relaxation(summary):

    arrangements = ["Random", "Cluster"]
    n_rows = len(arrangements)
    n_cols = len(NOISES_TO_PLOT)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.4 * n_cols, 5.0 * n_rows),
        sharex=True,
        squeeze=False,
    )

    # --------------------------------------------------------
    # Optional common energy y-limits within each arrangement
    # --------------------------------------------------------
    row_ylims = {}

    if SHARE_ENERGY_Y_WITHIN_ARRANGEMENT:
        for arrangement in arrangements:
            d = summary[
                (summary["arrangement"] == arrangement)
                &
                summary["mesh_type"].apply(
                    lambda x: any(
                        np.isclose(x, noise)
                        for noise in NOISES_TO_PLOT
                    )
                )
            ]

            if not d.empty:
                ymin = d["energy_mean"].min()
                ymax = d["energy_mean"].max()
                pad = 0.04 * (ymax - ymin)

                if pad == 0:
                    pad = 0.04 * max(abs(ymax), 1.0)

                row_ylims[arrangement] = (
                    ymin - pad,
                    ymax + pad,
                )

    # --------------------------------------------------------
    # Panels
    # --------------------------------------------------------
    for row, arrangement in enumerate(arrangements):

        cmap = CMAPS[arrangement]

        for col, noise in enumerate(NOISES_TO_PLOT):

            ax = axes[row, col]

            panel = summary[
                (summary["arrangement"] == arrangement)
                &
                np.isclose(summary["mesh_type"], noise)
            ].copy()

            if panel.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"No data\n{arrangement}, D={noise:g}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            # ----------------------------------------------
            # Main energy curves
            # ----------------------------------------------
            for pct in sorted(panel["pct_mutant"].unique()):

                g = (
                    panel[
                        np.isclose(
                            panel["pct_mutant"],
                            pct,
                        )
                    ]
                    .sort_values("time")
                )

                color = cmap(PCT_NORM(pct))

                ax.plot(
                    g["time"],
                    g["energy_mean"],
                    color=color,
                    linewidth=1.6,
                    alpha=0.95,
                )

            ax.set_title(
                rf"{arrangement}, $D={noise:g}$",
                fontsize=14,
            )

            ax.set_xlabel(
                "Time",
                fontsize=12,
            )

            if col == 0:
                ax.set_ylabel(
                    "Mean tissue energy",
                    fontsize=12,
                )

            if TIME_MIN is not None or TIME_MAX is not None:
                xmin = (
                    TIME_MIN
                    if TIME_MIN is not None
                    else panel["time"].min()
                )
                xmax = (
                    TIME_MAX
                    if TIME_MAX is not None
                    else panel["time"].max()
                )
                ax.set_xlim(xmin, xmax)

            if arrangement in row_ylims:
                ax.set_ylim(*row_ylims[arrangement])

            ax.tick_params(
                axis="both",
                which="both",
                direction="out",
                length=4,
                labelsize=11,
                width=1, bottom=True, left = True
            )

            sns.despine(ax=ax)

            # ----------------------------------------------
            # Inset: dE/dt
            # ----------------------------------------------
            axins = inset_axes(
                ax,
                width="46%",
                height="42%",
                loc="upper right",
                borderpad=1.1,
            )

            inset_values = []

            for pct in sorted(panel["pct_mutant"].unique()):

                g = (
                    panel[
                        np.isclose(
                            panel["pct_mutant"],
                            pct,
                        )
                    ]
                    .sort_values("time")
                )

                color = cmap(PCT_NORM(pct))

                axins.plot(
                    g["time"],
                    g["dE_dt"],
                    color=color,
                    linewidth=1.0,
                    alpha=0.85,
                )
                inset_values.extend(
                    g["dE_dt"]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                    .tolist()
                )

            # Zero = stationary energy.
            axins.axhline(
                0.0,
                linestyle="--",
                linewidth=1.0,
                color="black",
                alpha=0.75,
            )

            axins.set_ylim(
                *robust_symmetric_ylim(
                    inset_values,
                    INSET_PERCENTILES,
                )
            )
            # The inset is zoomed to the first 5000 time units.
            axins.set_xlim(0, 5000)
            axins.set_title(
                r"Relaxation rate",
                fontsize=9,
                pad=2,
            )

            axins.set_ylabel(
                r"$dE/dt$",
                fontsize=8,
            )

            axins.tick_params(
                axis="both",
                which="both",
                direction="out",
                length=2.5,
                labelsize=7,
                width=0.8, bottom = True, left = True
            )

            sns.despine(ax=axins)

    # --------------------------------------------------------
    # Shared mutant-fraction colorbars
    # --------------------------------------------------------
    fig.subplots_adjust(
        right=0.90,
        top=0.90,
        bottom=0.09,
        left=0.08,
        hspace=0.28,
        wspace=0.20,
    )

    # One colorbar per arrangement, matching the corresponding row.
    for row, arrangement in enumerate(arrangements):

        sm = mpl.cm.ScalarMappable(
            norm=PCT_NORM,
            cmap=CMAPS[arrangement],
        )
        sm.set_array([])

        # Position each colorbar beside its row.
        row_height = 0.34
        y0 = 0.54 if row == 0 else 0.10

        cax = fig.add_axes(
            [
                0.92,
                y0,
                0.014,
                row_height,
            ]
        )

        cbar = fig.colorbar(
            sm,
            cax=cax,
        )

        cbar.set_label(
            rf"$f_{{\rm mut}}$ - {arrangement}",
            fontsize=10,
        )

        cbar.ax.tick_params(
            labelsize=8,
        )

    fig.suptitle(
        "Tissue energy relaxation across mutant fractions",
        fontsize=16,
        y=0.97,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    svg_path = os.path.join(
        SAVE_DIR,
        f"{OUTPUT_BASENAME}.svg",
    )

    png_path = os.path.join(
        SAVE_DIR,
        f"{OUTPUT_BASENAME}.png",
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved: {svg_path}")
    print(f"Saved: {png_path}")

    plt.show()
    plt.close(fig)


# ============================================================
# RUN
# ============================================================

print("Loading Random energy data...")
df_random = load_energy_data("Random")

print("Loading Cluster energy data...")
df_cluster = load_energy_data("Cluster")

df = pd.concat(
    [
        df_random,
        df_cluster,
    ],
    ignore_index=True,
)

del df_random
del df_cluster
gc.collect()

df = restrict_time(df)

print("Calculating mean energy and dE/dt...")
summary = calculate_mean_energy_and_relaxation(df)

print("Generating figure...")
plot_energy_relaxation(summary)

print("")
print("============================================================")
print("ENERGY RELAXATION FIGURE FINISHED")
print("============================================================")
print(f"Saved in: {SAVE_DIR}")
print("")
