# ============================================================
# HETEROTYPIC INTERFACE FIGURE (main text Figs. 2 and 3)
# Heterotypic interface geometry and mutant clearance
# ============================================================
#
# A-B  Mutant clearance N_mut(t)/N0 vs time for two D values
# C    Total heterotypic interface L_het(t) vs time
# D    Heterotypic exposure per mutant L_het(t)/N_mut(t) vs time
# E    Dynamic scaling L_het(t)/L_het(0) vs q=N_mut(t)/N0
# F    Dynamic exposure scaling chi(t)/chi(0) vs q
#
# Panels C-F use DETAIL_NOISE. Run the script once with
# DETAIL_NOISE = NOISE_LOW and once with DETAIL_NOISE = NOISE_HIGH.
# Panels A-D use TIME_MAX for the x-axis limit.
# ============================================================

import os
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from pyprojroot import here
from vertex_lite.custom import load_data_v3 as load_data


# ============================================================
# SETTINGS -- EDIT THESE
# ============================================================

PCTS = [round(v, 3) for v in np.arange(0.025, 0.301, 0.025)]

# A-B: the two initial-disorder values shown side by side.
NOISE_LOW = 0.1
NOISE_HIGH = 0.5

# C-F: detailed mechanistic panels use ONE representative D.
# Change to NOISE_HIGH if you prefer the high-disorder condition.
DETAIL_NOISE = NOISE_LOW

# Final time shown ONLY on panels whose x-axis is Time (A-D).
TIME_MAX = 5000

# Dynamic-scaling settings (E-F).
Q_MIN = 0.05
N_Q_BINS = 20

# Read E-F in the physical clearance direction: q=1 -> q=Q_MIN.
REVERSE_Q_AXIS = True

# Optional uncertainty bands in the time-series panels A-D.
SHOW_SEM = True
SEM_ALPHA = 0.10

# Horizontal clearance guides in A-B.
CLEARANCE_GUIDES = [0.50, 0.05]

# Styling.
RANDOM_COLOR = "#709E0D"
CLUSTER_COLOR = "#A53379"

COLORS = {
    "random": RANDOM_COLOR,
    "cluster": CLUSTER_COLOR,
}

SAVE_DIR = here(
    "results/analysis/interface_scaling"
)
os.makedirs(SAVE_DIR, exist_ok=True)

OUTPUT_BASENAME = (
    f"interface_ABCDEF_Ddetail_{DETAIL_NOISE:g}_tmax_{TIME_MAX:g}"
)


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


def safe_sem(values):
    values = pd.Series(values).dropna()
    if len(values) <= 1:
        return np.nan
    return values.sem()


def make_fraction_cmap(base_color):
    return sns.blend_palette(
        [
            sns.light_palette(base_color, n_colors=8)[2],
            base_color,
        ],
        as_cmap=True,
    )


CMAPS = {
    "random": make_fraction_cmap(RANDOM_COLOR),
    "cluster": make_fraction_cmap(CLUSTER_COLOR),
}

PCT_NORM = mpl.colors.Normalize(vmin=min(PCTS), vmax=max(PCTS))


# ============================================================
# LOAD + NORMALISE DATA
# ============================================================


def load_interface_data(arrangement):
    if arrangement == "cluster":
        rpath = here("results/resultados_cluster")
    elif arrangement == "random":
        rpath = here("results/resultados_random")
    else:
        raise ValueError("arrangement must be 'random' or 'cluster'")

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
                "interface_perimeter",
                "n_alive_mutants",
            ],
        )

        if not d.empty:
            dfs.append(d)

    if len(dfs) == 0:
        raise ValueError(f"No interface.parquet data found for {arrangement}")

    df = pd.concat(dfs, ignore_index=True)
    df["arrangement"] = arrangement

    group_cols = [
        "arrangement",
        "mesh_type",
        "pct_mutant",
        "sim_id",
    ]

    # Remove duplicate snapshots at the same saved time if needed.
    df = (
        df.groupby(group_cols + ["time"], as_index=False)
        .agg(
            interface_perimeter=("interface_perimeter", "mean"),
            n_alive_mutants=("n_alive_mutants", "max"),
        )
        .sort_values(group_cols + ["time"])
        .reset_index(drop=True)
    )

    # Heterotypic exposure per remaining mutant.
    df["normalized_interface"] = np.where(
        df["n_alive_mutants"] > 0,
        df["interface_perimeter"] / df["n_alive_mutants"],
        np.nan,
    )

    # Initial values independently for every stochastic simulation.
    df["N0"] = (
        df.groupby(group_cols)["n_alive_mutants"]
        .transform("first")
        .astype(float)
    )

    df["L0"] = (
        df.groupby(group_cols)["interface_perimeter"]
        .transform("first")
        .astype(float)
    )

    df["chi0"] = (
        df.groupby(group_cols)["normalized_interface"]
        .transform("first")
        .astype(float)
    )

    # Remaining population fraction.
    df["q"] = np.where(
        df["N0"] > 0,
        df["n_alive_mutants"] / df["N0"],
        np.nan,
    )

    # Dynamic normalisations used in E-F.
    df["L_ratio"] = np.where(
        df["L0"] > 0,
        df["interface_perimeter"] / df["L0"],
        np.nan,
    )

    df["chi_ratio"] = np.where(
        df["chi0"] > 0,
        df["normalized_interface"] / df["chi0"],
        np.nan,
    )

    return df


# ============================================================
# TIME-SERIES SUMMARY: EACH SIMULATION HAS EQUAL WEIGHT
# ============================================================


def summarize_time_series(df, noise, arrangement):
    """
    Average stochastic realisations at each saved time for every f_mut.
    The simulation is the replicate: at a given time each sim_id contributes
    one value to the condition mean.
    """
    d = df[
        np.isclose(df["mesh_type"], noise)
        & (df["arrangement"] == arrangement)
        & (df["time"] <= TIME_MAX)
    ].copy()

    if d.empty:
        return pd.DataFrame()

    return (
        d.groupby(["pct_mutant", "time"], as_index=False)
        .agg(
            q_mean=("q", "mean"),
            q_sem=("q", safe_sem),
            L_mean=("interface_perimeter", "mean"),
            L_sem=("interface_perimeter", safe_sem),
            chi_mean=("normalized_interface", "mean"),
            chi_sem=("normalized_interface", safe_sem),
            n_sim=("sim_id", "nunique"),
        )
        .sort_values(["pct_mutant", "time"])
    )


# ============================================================
# q-BINNING FOR E-F
# ============================================================


def bin_one_simulation_in_q(g, q_edges):
    """
    Average one simulation inside logarithmic q bins. The initial point
    (q=1) is included in the highest-q bin.
    """
    d = g[
        np.isfinite(g["q"])
        & np.isfinite(g["L_ratio"])
        & np.isfinite(g["chi_ratio"])
        & (g["q"] > 0)
        & (g["q"] <= 1)
    ].copy()

    if d.empty:
        return pd.DataFrame()

    d = d.sort_values("time")

    d["q_bin"] = pd.cut(
        d["q"],
        bins=q_edges,
        include_lowest=True,
        labels=False,
    )

    out = (
        d.dropna(subset=["q_bin"])
        .groupby("q_bin", observed=True)
        .agg(
            q=("q", "mean"),
            L_ratio=("L_ratio", "mean"),
            chi_ratio=("chi_ratio", "mean"),
        )
        .reset_index()
    )
    out["is_initial"] = False
    return out


def build_binned_simulation_table(df):
    # Slightly above 1 so q=1 is definitely included.
    q_edges = np.geomspace(Q_MIN, 1.000001, N_Q_BINS + 1)

    rows = []
    group_cols = [
        "arrangement",
        "mesh_type",
        "pct_mutant",
        "sim_id",
    ]

    for keys, g in df.groupby(group_cols):
        binned = bin_one_simulation_in_q(g, q_edges)

        if binned.empty:
            continue

        for name, value in zip(group_cols, keys):
            binned[name] = value

        rows.append(binned)

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)



def summarize_binned_scaling(binned, noise, arrangement):
    d = binned[
        np.isclose(binned["mesh_type"], noise)
        & (binned["arrangement"] == arrangement)
    ].copy()

    if d.empty:
        return pd.DataFrame()

    return (
        d.groupby(
            ["pct_mutant", "q_bin", "is_initial"],
            as_index=False,
            observed=True,
        )
        .agg(
            q=("q", "mean"),
            L_ratio=("L_ratio", "mean"),
            L_ratio_sem=("L_ratio", safe_sem),
            chi_ratio=("chi_ratio", "mean"),
            chi_ratio_sem=("chi_ratio", safe_sem),
            n_sim=("sim_id", "nunique"),
        )
        .sort_values(["pct_mutant", "q"], ascending=[True, False])
    )


# ============================================================
# PLOTTING HELPERS
# ============================================================


def style_time_axis(ax):
    ax.set_xlim(0, TIME_MAX)

    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        length=4,
        labelsize=11,
        width=1,
        bottom=True,
        left=True,
        top=False,
        right=False,
    )

    ax.tick_params(axis="x", rotation=45)
    sns.despine(ax=ax, top=True, right=True)



def style_scaling_axis(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")

    if REVERSE_Q_AXIS:
        ax.set_xlim(1.02, Q_MIN)
    else:
        ax.set_xlim(Q_MIN, 1.02)

    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        length=4,
        labelsize=11,
        width=1,
        bottom=True,
        left=True,
        top=False,
        right=False,
    )
    ax.tick_params(axis="x", which="both", rotation=45)
    sns.despine(ax=ax, top=True, right=True)



def add_panel_label(ax, label):
    ax.text(
        -0.13,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
    )



def plot_condition_time_curve(ax, summary, pct, arrangement, ycol, semcol=None):
    g = summary[np.isclose(summary["pct_mutant"], pct)].sort_values("time")
    if g.empty:
        return

    color = CMAPS[arrangement](PCT_NORM(pct))
    x = g["time"].to_numpy(dtype=float)
    y = g[ycol].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return

    ax.plot(
        x,
        y,
        color=color,
        linewidth=1.7,
        alpha=0.95,
    )

    if SHOW_SEM and semcol is not None:
        sem_all = g[semcol].to_numpy(dtype=float)
        sem = sem_all[valid]
        sem = np.nan_to_num(sem, nan=0.0)
        ax.fill_between(
            x,
            y - sem,
            y + sem,
            color=color,
            alpha=SEM_ALPHA,
            linewidth=0,
        )


# ============================================================
# MAIN A-F FIGURE
# ============================================================


def make_interface_figure(df, binned):
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(13, 10),
    )

    axA, axB = axes[0]
    axC, axD = axes[1]
    axE, axF = axes[2]

    # --------------------------------------------------------
    # A-B: CLEARANCE FOR TWO D VALUES
    # --------------------------------------------------------
    for ax, noise in [(axA, NOISE_LOW), (axB, NOISE_HIGH)]:
        for arrangement in ["random", "cluster"]:
            summary = summarize_time_series(df, noise, arrangement)
            if summary.empty:
                continue

            for pct in PCTS:
                plot_condition_time_curve(
                    ax,
                    summary,
                    pct,
                    arrangement,
                    ycol="q_mean",
                    semcol="q_sem",
                )

        for yguide in CLEARANCE_GUIDES:
            ax.axhline(
                yguide,
                linestyle="--",
                color="0.25",
                linewidth=1.0,
                alpha=0.9,
            )

        ax.set_ylim(-0.02, 1.03)
        ax.set_title(rf"Mutant clearance, $D={noise:g}$", fontsize=13)
        ax.set_xlabel("Time", fontsize=12)
        style_time_axis(ax)

    axA.set_ylabel(
        r"Remaining mutant fraction $N_{\rm mut}(t)/N_0$",
        fontsize=12,
    )
    axB.set_ylabel("")
    
    # --------------------------------------------------------
    # C: TOTAL HETEROTYPIC INTERFACE VS TIME
    # D: EXPOSURE PER MUTANT VS TIME
    # --------------------------------------------------------
    for arrangement in ["random", "cluster"]:
        summary = summarize_time_series(df, DETAIL_NOISE, arrangement)
        if summary.empty:
            continue

        for pct in PCTS:
            plot_condition_time_curve(
                axC,
                summary,
                pct,
                arrangement,
                ycol="L_mean",
                semcol="L_sem",
            )
            plot_condition_time_curve(
                axD,
                summary,
                pct,
                arrangement,
                ycol="chi_mean",
                semcol="chi_sem",
            )

    axC.set_title(
        rf"Total heterotypic interface, $D={DETAIL_NOISE:g}$",
        fontsize=13,
    )
    axD.set_title(
        rf"Heterotypic exposure per mutant, $D={DETAIL_NOISE:g}$",
        fontsize=13,
    )
    axC.set_xlabel("Time", fontsize=12)
    axD.set_xlabel("Time", fontsize=12)
    axC.set_ylabel(r"$L_{\rm het}(t)$", fontsize=12)
    axD.set_ylabel(r"$\chi_{\rm het}(t)=L_{\rm het}(t)/N_{\rm mut}(t)$", fontsize=12)

    style_time_axis(axC)
    style_time_axis(axD)

    # --------------------------------------------------------
    # E-F: DYNAMIC SCALING AT DETAIL_NOISE
    # --------------------------------------------------------
    for arrangement in ["random", "cluster"]:
        summary = summarize_binned_scaling(
            binned,
            DETAIL_NOISE,
            arrangement,
        )
        if summary.empty:
            continue

        for pct in PCTS:
            g = summary[
                np.isclose(summary["pct_mutant"], pct)
            ].sort_values("q", ascending=False)

            if g.empty:
                continue

            color = CMAPS[arrangement](PCT_NORM(pct))

            axE.plot(
                g["q"],
                g["L_ratio"],
                color=color,
                linewidth=1.7,
                alpha=0.95,
            )

            axF.plot(
                g["q"],
                g["chi_ratio"],
                color=color,
                linewidth=1.7,
                alpha=0.95,
            )

    qref = np.geomspace(Q_MIN, 1.0, 400)

    # E references.
    axE.plot(
        qref,
        qref ** 0.5,
        "--",
        color="black",
        linewidth=1.6,
        label=r"Compact 2D: $\beta=1/2$",
    )
    axE.plot(
        qref,
        qref,
        ":",
        color="0.45",
        linewidth=1.5,
        label=r"Independent defects: $\beta=1$",
    )

    # F references.
    axF.plot(
        qref,
        qref ** (-0.5),
        "--",
        color="black",
        linewidth=1.6,
        label=r"Compact 2D: $\alpha=-1/2$",
    )
    axF.axhline(
        1.0,
        linestyle=":",
        color="0.45",
        linewidth=1.5,
        label=r"Independent defects: $\alpha=0$",
    )

    axE.set_title(
        rf"Dynamic interface scaling, $D={DETAIL_NOISE:g}$",
        fontsize=13,
    )
    axF.set_title(
        rf"Dynamic exposure scaling, $D={DETAIL_NOISE:g}$",
        fontsize=13,
    )

    axE.set_xlabel(
        r"Remaining mutant fraction $q=N_{\rm mut}(t)/N_0$",
        fontsize=12,
    )
    axF.set_xlabel(
        r"Remaining mutant fraction $q=N_{\rm mut}(t)/N_0$",
        fontsize=12,
    )

    axE.set_ylabel(r"$L_{\rm het}(t)/L_{\rm het}(0)$", fontsize=12)
    axF.set_ylabel(r"$\chi_{\rm het}(t)/\chi_{\rm het}(0)$", fontsize=12)

    # Ranges chosen to include the values seen in the current simulations.
    axE.set_ylim(8e-4, 1.15)
    axF.set_ylim(8e-3, 11.0)

    style_scaling_axis(axE)
    style_scaling_axis(axF)

    axE.legend(frameon=False, fontsize=8, loc="lower left")
    axF.legend(frameon=False, fontsize=8, loc="upper left")

    # --------------------------------------------------------
    # PANEL LABELS
    # --------------------------------------------------------
    for ax, label in zip(
        [axA, axB, axC, axD, axE, axF],
        ["A", "B", "C", "D", "E", "F"],
    ):
        add_panel_label(ax, label)

    # --------------------------------------------------------
    # GLOBAL RANDOM / CLUSTER LEGEND
    # --------------------------------------------------------
    arrangement_handles = [
        mpl.lines.Line2D(
            [], [],
            color=RANDOM_COLOR,
            linewidth=4,
            label="Random",
        ),
        mpl.lines.Line2D(
            [], [],
            color=CLUSTER_COLOR,
            linewidth=4,
            label="Cluster",
        ),
    ]

    fig.legend(
        handles=arrangement_handles,
        loc="upper center",
        bbox_to_anchor=(0.49, 0.995),
        ncol=2,
        frameon=False,
        fontsize=12,
    )

    # --------------------------------------------------------
    # TWO SHARED f_mut COLORBARS
    # --------------------------------------------------------
    sm_random = mpl.cm.ScalarMappable(
        norm=PCT_NORM,
        cmap=CMAPS["random"],
    )
    sm_random.set_array([])

    sm_cluster = mpl.cm.ScalarMappable(
        norm=PCT_NORM,
        cmap=CMAPS["cluster"],
    )
    sm_cluster.set_array([])

    # Leave room on the right for the two colorbars.
    fig.subplots_adjust(
        left=0.09,
        right=0.84,
        top=0.955,
        bottom=0.07,
        hspace=0.42,
        wspace=0.28,
    )

    cax_random = fig.add_axes([0.865, 0.16, 0.013, 0.72])
    cbar_random = fig.colorbar(sm_random, cax=cax_random)
    cbar_random.set_label(r"$f_{\rm mut}$ - Random", fontsize=10)
    cbar_random.ax.tick_params(labelsize=9, direction="out", length=3)

    cax_cluster = fig.add_axes([0.915, 0.16, 0.013, 0.72])
    cbar_cluster = fig.colorbar(sm_cluster, cax=cax_cluster)
    cbar_cluster.set_label(r"$f_{\rm mut}$ - Cluster", fontsize=10)
    cbar_cluster.ax.tick_params(labelsize=9, direction="out", length=3)

    # --------------------------------------------------------
    # TITLE + SAVE
    # --------------------------------------------------------
    fig.suptitle(
        "Heterotypic interface geometry and mutant clearance",
        fontsize=16,
        y=0.978,
    )

    svg_path = os.path.join(SAVE_DIR, OUTPUT_BASENAME + ".svg")
    png_path = os.path.join(SAVE_DIR, OUTPUT_BASENAME + ".png")

    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    print(f"Saved: {svg_path}")
    print(f"Saved: {png_path}")

    plt.show()
    plt.close(fig)


# ============================================================
# RUN
# ============================================================

print("Loading random simulations...")
df_random = load_interface_data("random")

print("Loading cluster simulations...")
df_cluster = load_interface_data("cluster")

df = pd.concat(
    [df_random, df_cluster],
    ignore_index=True,
)

del df_random
del df_cluster
gc.collect()

# Check requested D values exist.
available_noises = np.asarray(sorted(df["mesh_type"].dropna().unique()), dtype=float)
for requested_noise in [NOISE_LOW, NOISE_HIGH, DETAIL_NOISE]:
    if not np.any(np.isclose(available_noises, requested_noise)):
        raise ValueError(
            f"Requested D={requested_noise:g} is not present. "
            f"Available D values: {available_noises.tolist()}"
        )

print("Binning each independent simulation in q for panels E-F...")
binned = build_binned_simulation_table(df)

print("Generating A-F interface figure...")
make_interface_figure(df, binned)

print("")
print("============================================================")
print("A-F INTERFACE FIGURE FINISHED")
print("============================================================")
print(f"A-B disorder values: D={NOISE_LOW:g}, D={NOISE_HIGH:g}")
print(f"C-F detail disorder: D={DETAIL_NOISE:g}")
print(f"Time panels A-D shown until t={TIME_MAX:g}")
print(f"Saved in: {SAVE_DIR}")
print("")
