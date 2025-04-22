#!/usr/bin/env python3
# leaderboard.py
# ──────────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

# ── 0) MATPLOTLIB BACK‑END (avoid Wayland gripe) ─────────────────
# If you only need the PNG file and no pop‑up window, uncomment:
# import matplotlib
# matplotlib.use("Agg")          # headless rendering

# ── 1) READ YOUR CSV ──────────────────────────────────────────────
csv_path = "results.csv"
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"Cannot find {csv_path!r} in {os.getcwd()}")

results = pd.read_csv(csv_path)
results_full = results
results = results.sort_values("MSE", ascending=True)
results = results[:48]

# make sure MSE is truly numeric (drops commas / bad chars → NaN)
results["MSE"] = pd.to_numeric(results["MSE"], errors="coerce")

Models = results["Model"]        # keep as Series for convenience

# ── 2) SPLIT MODEL NAMES INTO COMPONENTS ─────────────────────────
def split_model_name(name: str) -> pd.Series:
    """
    'small_layers_left_pyramid_250_0.0005_AdamW.pt'
        → size, shape, epochs, lr, optimizer
    """
    clean  = name.rsplit(".", 1)[0]     # strip .pt
    parts  = clean.split("_")

    size = parts[0]
    idx_epochs = next(i for i, p in enumerate(parts) if p.isdigit())
    shape      = "_".join(parts[1:idx_epochs])
    epochs     = int(parts[idx_epochs])
    lr         = float(parts[idx_epochs + 1])
    optimizer  = parts[idx_epochs + 2]

    return pd.Series(
        [size, shape, epochs, lr, optimizer],
        index=["size", "shape", "epochs", "lr", "optimizer"],
    )

parsed_cols = Models.apply(split_model_name)

# ── 3) BUILD THE TABLE DATAFRAME ─────────────────────────────────
table_df = pd.concat([parsed_cols, results[["MSE"]]], axis=1)

# order best→worst and give each row a 1‑based index
table_df = (
    table_df
    .sort_values("MSE", ascending=True)
    .reset_index(drop=True)
    .assign(rank=lambda df: np.arange(1, len(df)+1))   # ➊
    .loc[:, ["rank", *table_df.columns]]               # ➋ put rank first
)

# ── 4) FIGURE & COLOUR MAP SET‑UP ────────────────────────────────
n_rows, n_cols = table_df.shape
fig_w, fig_h   = 2.0 * n_cols, 0.45 * (n_rows + 1)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.axis("off")

norm = mpl.colors.Normalize(vmin=0.46,
                            vmax=2.18)
cmap = mpl.cm.viridis           # use mpl.cm.viridis_r to invert

# ── 5) CONSTRUCT THE TABLE ───────────────────────────────────────
header     = list(table_df.columns)
cells      = table_df.values.tolist()
table_data = [header] + cells

tbl = ax.table(cellText=table_data, cellLoc="center", loc="center")

# turn *off* Matplotlib’s automatic down‑scaling **before** you touch cells
tbl.auto_set_font_size(False)

for (row, col), cell in tbl.get_celld().items():
    if row == 0:                                      # ── header
        cell.set_text_props(weight="bold", fontsize=14)   # larger header
        cell.set_facecolor("#dce6f2")
    else:                                             # ── data rows
        cell.set_fontsize(12)                             # body text
        if row % 2 == 0:
            cell.set_facecolor("#f6f6f6")                 # zebra stripe

        if col == n_cols - 1:                             # MSE column
            text_obj = cell.get_text()
            mse_val  = float(text_obj.get_text())
            text_obj.set_text(f"{mse_val:.3f}")           # 3‑dp
            text_obj.set_color("white")                   # white font
            cell.set_facecolor(cmap(norm(mse_val)))

    cell.set_edgecolor("0.8")
    cell.set_linewidth(0.5)

tbl.scale(1, 1.4)
plt.tight_layout()

# colour‑bar
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=ax, shrink=0.6, pad=0.02)
cbar.set_label("MSE")

# comment out the next line if you only want the file,
# or if matplotlib.use('Agg') is active
plt.show()
