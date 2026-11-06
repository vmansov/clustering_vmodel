import dill
import os
import numpy as np
from vertex_lite import custom, plotting, coloring
import matplotlib.pyplot as plt
import pyprojroot.here as here

parent_dir = here()
# Snapshot times: random (Figs. 1A, 4A, 5A, 7A) and cluster (Figs. 1B, 4B, 5B, 7B).
# 14980 is the last saved frame (t_max = 15000 with a saving skip of 20).
times = [[0, 50, 5000, 14980], [0, 500, 5000, 14980]]
noise = 0.25
pct = 0.25
sim = 0


## DEFINE FUNCTION TO CREATE FIGURES FOR SNAPSHOTS
def main_graph(noise, pct, sim, parent_dir, time, cluster=False):
    os.makedirs(os.path.join(parent_dir, f"results/new_snapshots/{'cluster' if cluster else 'rand'}"), exist_ok=True)

    with open(os.path.join(parent_dir, "results/resultados_cluster" if cluster else "results/resultados_random", f"noise_{noise}", f"pct_{pct}", f"sim_{sim}", "history.dill"), "rb") as f:
        history = dill.load(f)
    for t in time:
        real_time = int(t/20)
        cells = history[real_time]

        cells.properties['shape_index'] = [custom.shape_index(cells, i, normalize=False, max_shape_index=5.0) for i in range(len(cells.mesh.area))]
        cells.properties['n_lados'] = np.array([len(cells.mesh.boundary(i)) for i in range(len(cells))])
        cells.properties['teff'] = [custom.calculate_t_eff(cells, i) for i in range(len(cells))]
        values = cells.properties['shape_index']

        ## Plot Parent Group (Fig. 1)
        cells.properties['color'] = coloring.definecolors_manual(cells, property_name='parent_group')
        fig = plotting.draw(cells)
        fig.savefig(os.path.join(parent_dir, f"results/new_snapshots/{'cluster' if cluster else 'rand'}", f"parent_{noise}_{'cluster' if cluster else 'rand'}_pct_{pct}_sim_{sim}_time_{t}.svg"))
        plt.close(fig)
        ## Plot Shape Index (Fig. 5)
        cells.properties['color'] = coloring.values_to_hex_colors2(values, cmap_name='custom', fixed_vmin=3.72, fixed_vmax=3.9)
        fig = plotting.draw(cells)
        fig.savefig(os.path.join(parent_dir, f"results/new_snapshots/{'cluster' if cluster else 'rand'}", f"shape_{noise}_{'cluster' if cluster else 'rand'}_pct_{pct}_sim_{sim}_time_{t}.svg"))
        plt.close(fig)
        ## Plot number of neighbours (Fig. 7)
        cells.properties['color'] = coloring.definecolors_fixed_1_10_center6(cells, property_name='n_lados', cmap_name='BrBG')
        fig = plotting.draw(cells)
        fig.savefig(os.path.join(parent_dir, f"results/new_snapshots/{'cluster' if cluster else 'rand'}", f"lados_{noise}_{'cluster' if cluster else 'rand'}_pct_{pct}_sim_{sim}_time_{t}.svg"))
        plt.close(fig)

        ## Plot T_eff (Fig. 4)
        cells.properties['color'] = coloring.values_to_hex_colors2(cells.properties['teff'], cmap_name='gnuplot', fixed_vmin=0.25, fixed_vmax=1, limit_colors=False)
        fig = plotting.draw(cells)
        fig.savefig(os.path.join(parent_dir, f"results/new_snapshots/{'cluster' if cluster else 'rand'}", f"teff_{noise}_{'cluster' if cluster else 'rand'}_pct_{pct}_sim_{sim}_time_{t}.svg"))
        plt.close(fig)


## CREATE FIGURES
main_graph(noise=noise, pct=pct, sim=sim, parent_dir=parent_dir, time=times[0], cluster=False)
main_graph(noise=noise, pct=pct, sim=sim, parent_dir=parent_dir, time=times[1], cluster=True)
