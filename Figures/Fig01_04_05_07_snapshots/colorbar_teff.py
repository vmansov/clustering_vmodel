# T_eff colour bar used in Fig. 4 (same scale as the T_eff snapshots in snapshots.py)
import os
import matplotlib.pyplot as plt
import pyprojroot.here as here
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib import colormaps

fixed_vmin = 0.25
fixed_vmax = 1.0
cmap_name = "gnuplot"

cmap = colormaps[cmap_name]
norm = Normalize(vmin=fixed_vmin, vmax=fixed_vmax)

fig, ax = plt.subplots(figsize=(1.2, 4))
fig.subplots_adjust(left=0.35, right=0.75)

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

colorbar = fig.colorbar(sm, cax=ax)
colorbar.set_label(r"$T_{\mathrm{eff}}$")

save_dir = here("results/new_snapshots")
os.makedirs(save_dir, exist_ok=True)

fig.savefig(
    os.path.join(save_dir, "colorbar_teff.pdf"),
    bbox_inches="tight",
    transparent=False,
)

plt.close(fig)
