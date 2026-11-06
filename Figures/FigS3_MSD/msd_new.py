import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from matplotlib import colors
import pyprojroot.here as here


# ============================================================
# SETTINGS
# ============================================================

# Cumulative MSD relative to t = 0 (MSD_0 in the SI)
timescale = "total"

# Mean over the selected cells (column name in the stats parquet)
toplot = "msd_global"

# The SI figure uses the mutant-neighbourhood (local) cells only
local = True

# Choose the two initial mesh distortions
NOISES_TO_PLOT = [
    0.1,
    0.5
]

# None = plot complete trajectory
max_time = None

# MSD normally makes most sense in log-log for this analysis
logarithmic = True

# SEM bands are not shown in the SI figure
show_sem = False


RANDOM_COLOR = "#709E0D"
CLUSTER_COLOR = "#A53379"


# ============================================================
# PATHS
# ============================================================

datapath = here(
    f"results/analysis/msd/msd_stats/{timescale}"
)

save_dir = here(
    f"results/analysis/msd/msd_plots_by_noise/{timescale}"
)

os.makedirs(
    save_dir,
    exist_ok=True
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
    }
)


# ============================================================
# COLOUR MAP
#
# Light = low mutant fraction
# Dark  = high mutant fraction
# ============================================================

def make_fraction_cmap(base_color):

    return sns.blend_palette(
        [
            sns.light_palette(
                base_color,
                n_colors=8
            )[2],
            base_color
        ],
        as_cmap=True
    )


cmap_random = make_fraction_cmap(
    RANDOM_COLOR
)

cmap_cluster = make_fraction_cmap(
    CLUSTER_COLOR
)


# ============================================================
# PLOT FUNCTION
# ============================================================

def plot_msd_fixed_noises(
    stats_random,
    stats_cluster,
    local=False,
    toplot="msd_global",
    noises=(0.15, 0.65),
    max_time=None,
    logarithmic=True,
    show_sem=False
):

    """
    Create one figure with one panel per fixed mesh distortion D.

    Within every panel:
        - all mutant fractions are shown
        - Random uses a green gradient
        - Cluster uses a magenta gradient
        - darker colour = larger f_mut
    """

    toplot_sem = f"{toplot}_sem"


    # --------------------------------------------------------
    # Determine common mutant-fraction range
    # --------------------------------------------------------

    all_pcts = sorted(
        set(
            stats_random[
                "pct_mutant"
            ].unique()
        ).union(
            stats_cluster[
                "pct_mutant"
            ].unique()
        )
    )


    if len(all_pcts) == 0:

        print(
            "No mutant percentages found."
        )

        return None


    pct_min = min(
        all_pcts
    )

    pct_max = max(
        all_pcts
    )


    norm_pct = colors.Normalize(
        vmin=pct_min,
        vmax=pct_max
    )


    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        len(noises),
        figsize=(
            13,
            5.3
        ),
        sharex=True,
        sharey=True
    )


    if len(noises) == 1:

        axes = [
            axes
        ]


    # ========================================================
    # LOOP OVER THE TWO FIXED NOISES
    # ========================================================

    for ax, noise in zip(
        axes,
        noises
    ):


        # ----------------------------------------------------
        # RANDOM + CLUSTER
        # ----------------------------------------------------

        for (
            stats,
            arrangement,
            cmap
        ) in [

            (
                stats_random,
                "Random",
                cmap_random
            ),

            (
                stats_cluster,
                "Cluster",
                cmap_cluster
            )

        ]:


            # Select this noise only
            data_noise = stats[
                np.isclose(
                    stats[
                        "mesh_type"
                    ],
                    noise
                )
            ].copy()


            # =================================================
            # ALL MUTANT FRACTIONS
            # =================================================

            for pct in all_pcts:


                d = data_noise[
                    np.isclose(
                        data_noise[
                            "pct_mutant"
                        ],
                        pct
                    )
                ].copy()


                if d.empty:
                    continue


                d = d.sort_values(
                    "time"
                )


                if max_time is not None:

                    d = d[
                        d[
                            "time"
                        ]
                        <=
                        max_time
                    ]


                # Logarithmic x-axis cannot show time = 0
                if logarithmic:

                    d = d[
                        d[
                            "time"
                        ]
                        >
                        0
                    ]


                if d.empty:
                    continue


                x = d[
                    "time"
                ].to_numpy()


                y = d[
                    toplot
                ].to_numpy()


                # Protect log scale
                if logarithmic:

                    y = np.clip(
                        y,
                        1e-12,
                        None
                    )


                color = cmap(
                    norm_pct(
                        pct
                    )
                )


                # ------------------------------------------------
                # Mean line
                # ------------------------------------------------

                ax.plot(
                    x,
                    y,
                    color=color,
                    linewidth=1.6,
                    alpha=0.95
                )


                # ------------------------------------------------
                # SEM
                # ------------------------------------------------

                if (
                    show_sem
                    and
                    toplot_sem
                    in d.columns
                ):


                    sem = d[
                        toplot_sem
                    ].to_numpy()


                    low = np.maximum(
                        y
                        -
                        np.abs(
                            sem
                        ),
                        1e-12
                        if logarithmic
                        else
                        -np.inf
                    )


                    high = (
                        y
                        +
                        np.abs(
                            sem
                        )
                    )


                    ax.fill_between(
                        x,
                        low,
                        high,
                        color=color,
                        alpha=0.08,
                        linewidth=0
                    )


        # ====================================================
        # AXIS FORMAT
        # ====================================================

        if logarithmic:

            ax.set_xscale(
                "log"
            )

            ax.set_yscale(
                "log"
            )


        ax.set_title(
            rf"$D={noise:g}$",
            fontsize=15
        )


        ax.set_xlabel(
            r"Time",
            fontsize=13
        )


        ax.tick_params(
            axis="both",
            which="both",
            direction="out",
            length=4,
            width=1,
            labelsize=11, bottom = True, left = True
        )


        ax.spines[
            "top"
        ].set_visible(
            False
        )


        ax.spines[
            "right"
        ].set_visible(
            False
        )


    # ========================================================
    # Y LABEL
    # ========================================================

    axes[
        0
    ].set_ylabel(
        (
            "Local MSD"
            if local
            else
            "Global MSD"
        ),
        fontsize=13
    )


    # ========================================================
    # RANDOM / CLUSTER LEGEND
    # ========================================================

    random_handle = mpl.lines.Line2D(
        [],
        [],
        color=RANDOM_COLOR,
        linewidth=4,
        label="Random"
    )


    cluster_handle = mpl.lines.Line2D(
        [],
        [],
        color=CLUSTER_COLOR,
        linewidth=4,
        label="Cluster"
    )


    fig.legend(
        handles=[
            random_handle,
            cluster_handle
        ],
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.015
        ),
        ncol=2,
        frameon=False,
        fontsize=13
    )


    # ========================================================
    # COLOURBARS
    # ========================================================

    sm_random = mpl.cm.ScalarMappable(
        norm=norm_pct,
        cmap=cmap_random
    )

    sm_random.set_array(
        []
    )


    sm_cluster = mpl.cm.ScalarMappable(
        norm=norm_pct,
        cmap=cmap_cluster
    )

    sm_cluster.set_array(
        []
    )


    # Leave room on the right
    fig.subplots_adjust(
        right=0.84,
        top=0.88,
        bottom=0.16,
        wspace=0.12
    )


    # Random colourbar
    cax_random = fig.add_axes(
        [
            0.865,
            0.19,
            0.014,
            0.66
        ]
    )


    cbar_random = fig.colorbar(
        sm_random,
        cax=cax_random
    )


    cbar_random.set_label(
        r"$f_{\rm mut}$ — Random",
        fontsize=10
    )


    # Cluster colourbar
    cax_cluster = fig.add_axes(
        [
            0.91,
            0.19,
            0.014,
            0.66
        ]
    )


    cbar_cluster = fig.colorbar(
        sm_cluster,
        cax=cax_cluster
    )


    cbar_cluster.set_label(
        r"$f_{\rm mut}$ — Cluster",
        fontsize=10
    )


    # ========================================================
    # TITLE
    # ========================================================

    fig.suptitle(
        (
            "Wild-type cell mean-squared displacement"
            +
            (
                " — local"
                if local
                else
                " — whole tissue"
            )
        ),
        fontsize=16,
        y=0.96
    )


    return fig


# ============================================================
# RUN
# ============================================================

local_str = "local" if local else "global"

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

stats_random = pd.read_parquet(
    f"{datapath}/msd_stats_{local_str}_random.parquet"
)

stats_cluster = pd.read_parquet(
    f"{datapath}/msd_stats_{local_str}_cluster.parquet"
)


# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------

fig = plot_msd_fixed_noises(
    stats_random=stats_random,
    stats_cluster=stats_cluster,
    local=local,
    toplot=toplot,
    noises=NOISES_TO_PLOT,
    max_time=max_time,
    logarithmic=logarithmic,
    show_sem=show_sem
)


if fig is not None:

    filename = (
        f"{toplot}_"
        f"{local_str}_"
        f"fixed_noises_"
        f"{NOISES_TO_PLOT[0]:g}_"
        f"{NOISES_TO_PLOT[1]:g}_"
        f"{'log' if logarithmic else 'linear'}"
        f".svg"
    )

    fig.savefig(
        os.path.join(
            save_dir,
            filename
        ),
        format="svg",
        bbox_inches="tight"
    )

    print(
        f"Saved: {filename}"
    )

    plt.show()

    plt.close(
        fig
    )
