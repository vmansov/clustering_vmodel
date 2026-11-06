# Scripts that reproduce the paper figures

This folder contains the scripts that produce every figure of the main text and the Supplementary Information (SI), plus the supplementary videos. Each script only contains the code needed for its figure. Some comments inside the code may still be in Spanish.

All final figures were assembled in **Inkscape** from SVG panels produced by these scripts. Panel letters, titles, legends and some colour bars were added or edited by hand.

---

## Main text

| Fig. | PDF in the manuscript | Script(s) in this folder | Panels produced (inside the project) |
|---|---|---|---|
| 1 | `PARENTGROUP_PAPER.pdf` | `Fig01_04_05_07_snapshots/snapshots.py` | `results/new_snapshots/{rand,cluster}/parent_0.25_*_pct_0.25_sim_0_time_{0,50,5000}` (random) and `{0,500,5000}` (cluster) |
| 2 | `HETEROTYPIC_LENGTH_PAPER_V2.pdf` | `Fig02_03_heterotypic_interface/figures_interface.py` | `results/analysis/interface_scaling/interface_ABCDEF_Ddetail_{0.1,0.5}_tmax_5000.svg` |
| 3 | `EXPONENT_NORMALISED_HETEROTYPIC_LENGTH.pdf` | same script as Fig. 2 | panel F of the two outputs above |
| 4 | `TEFF_PANELAB.pdf` | `Fig01_04_05_07_snapshots/snapshots.py` + `colorbar_teff.py` | `results/new_snapshots/.../teff_*.svg` + `results/new_snapshots/colorbar_teff.pdf` |
| 5 | `SHAPEINDEX_PANELAB_PAPER.pdf` | `Fig01_04_05_07_snapshots/snapshots.py` | `results/new_snapshots/.../shape_*.svg` |
| 6 | `SHAPE_INDEX_PAPER.pdf` | `Fig06_shape_index/si_plot.ipynb` (heatmaps) + `si_time.ipynb` (time curves) | `results/si/plots/heatmaps/heatmap_{random,cluster}_time_14980_shape_{local,global}.svg` and `results/si/plots/si_time/si_pct_{0.05,0.25}_*_5000.svg` |
| 7 | `NNEIGHBOUR_PAPER.pdf` | `Fig01_04_05_07_snapshots/snapshots.py` | `results/new_snapshots/.../lados_*.svg` |
| 8 | `nneighbours_noise_0_25_pct_0_25.pdf` | `Fig08_fraccion_poligonos/pol_fract.ipynb` | `results/nlados_time_plots/noise_0.25/pct_0.25/polygon_fraction_pct_0.25_noise_0.25_{local,global}_{uniform,cluster}.svg` |
| 9 | `DEV_HEX_PAPER.pdf` | `Fig09_desviacion_hexagonal/pol_plot.ipynb` (heatmaps) + `pol_time_plot.ipynb` (time curves) | `results/polygons/plots/pol_heatmaps/heatmap_*_time_14980_hex_*.svg` and `results/polygons/plots/pol_time/hex_deviation_pct_*_5000.svg` |
| 10 | `TEFF_IN_TIME_PAPER.pdf` | `Fig10_teff_tiempo/teff_time_joined.ipynb` | `results/analysis/teff/plots/teff_global_noise_{0.1,0.5}_{local,global}_overlay_cluster_vs_random_linear.svg` |

## Supplementary Information

| Fig. | PDF | Script(s) in this folder | Output inside the project |
|---|---|---|---|
| S1 | `ENERGY_PAPER.pdf` | `FigS1_energy/energy_figures.py` | `results/analysis/energy_relaxation/energy_plateau_with_relaxation_insets.svg` |
| S2 | `EXTRUSION_RATIO_PAPER.pdf` | `FigS2_extrusion_times/ratio_Extrusion.ipynb` | `results/supplementary_figures/heatmap_extrusion_time_{random,cluster}_{5,50}pct.svg` and `heatmap_extrusion_time_ratio_cluster_random_{5,50}pct.svg` |
| S3 | `MSD_PAPER.pdf` | `FigS3_MSD/mean_square_displacement.ipynb` (computes) → `msd_new.py` (plots) | `results/analysis/msd/msd_stats/total/msd_stats_local_{random,cluster}.parquet` → `results/analysis/msd/msd_plots_by_noise/total/msd_global_local_fixed_noises_0.1_0.5_log.svg` |
| S4 | `hist_final.pdf` | `FigS4_histograms/hist.ipynb` | `results/histograms/Hist_{uniform,Cluster}_{Local,Global}_pct[0.05, 0.25]_noise[0.2, 0.7].svg` |

## Supplementary videos S1–S8

`Videos_S1-S8/VIDEO.ipynb` produces all eight videos as `results/videos/video_{arrangement}_{tocolor}_pct_0.25_noise_0.25.mp4`:

| Video | `arrangement` | `tocolor` | Colouring |
|---|---|---|---|
| S1 | random | `mut` | mutant / wild-type |
| S2 | cluster | `mut` | mutant / wild-type |
| S3 | random | `tef_color` | effective edge tension |
| S4 | cluster | `tef_color` | effective edge tension |
| S5 | random | `si` | shape index |
| S6 | cluster | `si` | shape index |
| S7 | random | `nl` | number of neighbours |
| S8 | cluster | `nl` | number of neighbours |

---

## Step 0: simulation data

All figure scripts read from `results/resultados_random/` and `results/resultados_cluster/`, i.e. `noise_*/pct_*/sim_*/{cells,global,mutant,extrusion,interface}.parquet`, and write their outputs inside `results/`.

- **`para_sim.py`** (random) and **`para_sim_cluster.py`** (cluster), in the repository root, generate the full grid: D = 0–0.7, f_mut = 0–0.30 and 10 simulations per condition, up to t = 15000.
  - Frames are saved every 20 time units (`skip = 20`), so the last saved frame is t = 14980. This is the time used for the heatmaps and histograms ("t = 15000" in the manuscript).
  - The snapshots (Figs. 1, 4, 5 and 7) and the videos also need `history.dill` for `noise_0.25/pct_0.25/sim_0`. set `save_dill=False` to `True` (and `compress=False`) for that condition when running the simulation scripts.

## How to run

1. Install the package from the repository root (`pip install -e .`). All scripts depend on the `vertex_lite` package.
2. The figure scripts also need `numpy`, `pandas`, `pyarrow` (to read the parquet files), `matplotlib`, `seaborn`, `pyprojroot`, `dill` and `jupyter`. The videos also need `ffmpeg` (with `libx264`) installed on the system.
3. Run the scripts **from the repository root**. Most of them locate folders with `pyprojroot.here()`, and `pol_plot.ipynb` and `pol_time_plot.ipynb` use relative paths that assume the working directory is the repository root. If you run them from this folder, they will not find the data. 

4. Run `FigS3_MSD/mean_square_displacement.ipynb` before `FigS3_MSD/msd_new.py`, since the second one reads the parquet files written by the first one.

All outputs (`.svg`, `.pdf`, `.png`, `.mp4`, `.parquet`) are written inside `results/` and are ignored by git (see `.gitignore`).

## Settings needed to reproduce the manuscript figures exactly

- **Figs. 2 and 3 (`figures_interface.py`)**: run it **twice**, once with `DETAIL_NOISE = NOISE_LOW` (0.1) and once with `DETAIL_NOISE = NOISE_HIGH` (0.5).
  - Fig. 2A and 2B: panels A and B of either output.
  - Fig. 2C and 2D (L_het insets): panel C of the D = 0.1 output and panel C of the D = 0.5 output, respectively.
  - Fig. 2E and 2F (χ_het): panel D of each output.
  - Fig. 3A and 3B: panel F of each output. In the final PDF, the α legend was changed to β in Inkscape.
  - Panel E is produced by the script but not used in the manuscript.
- **Fig. S2 (`ratio_Extrusion.ipynb`)**: run all cells. Top row of the figure: the `50pct` outputs (t_50%); bottom row: the `5pct` outputs (t_5%). The ratio uses the brown palette `#7A4B41`.
- **Colour bars and legends**:
  - T_eff colour bar (Fig. 4): `colorbar_teff.py` produces `colorbar_teff.pdf` (gnuplot, 0.25–1).
  - Shape-index colour bar (Figs. 5 and 6): taken from the heatmaps produced by `si_plot.ipynb` (TwoSlopeNorm 3.72 / 3.81 / 3.90).
  - "Number of neighbours" polygon legend (Figs. 7 and 8): **no script generates it**. It was made with matplotlib and finished in Inkscape.
- **Snapshots (`snapshots.py`)**: the script also produces t = 14980 (last saved frame), which is not used in the manuscript.

