import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 24,          # base size for everything
    "axes.titlesize": 28,     # title font
    "axes.labelsize": 26,     # x‑/y‑label font
    "xtick.labelsize": 24,    # tick labels
    "ytick.labelsize": 24,
    "axes.linewidth": 3 
})

results = pd.read_csv("results.csv")

summary = (
    results[['MSE']]            # keep just the one column
        .describe()             # stats for MSE only
        .loc[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        .T.round(3)             # transpose so MSE is the row label
)

gap = .2                          # distance you want between boxes
pos = [1]  # just one position          # 1 → 1.35 → 1.70

fig, (ax_box, ax_tbl) = plt.subplots(
    2, 1,
    figsize=(8, 6),
    gridspec_kw={
        'height_ratios': [3, 1],
        'hspace': 0.6     # ← increase from 0.25 to 0.6 (or more)
    }
)

# ----- boxplot -----
ax_box.boxplot(
    results['MSE'],
    vert=False,
    positions=[1],  # explicitly set the position
    boxprops     =dict(linewidth=3),
    whiskerprops =dict(linewidth=3),
    capprops     =dict(linewidth=3),
    medianprops  =dict(linewidth=3),
    flierprops   =dict(marker='o', markersize=12,
                       markerfacecolor='none',
                       markeredgecolor='black',
                       markeredgewidth=2)
)
ax_box.set_xlabel('MSE')
ax_box.set_yticks([1])
ax_box.set_yticklabels([""])
ax_box.set_ylabel('All Models', labelpad=15)
ax_box.tick_params(axis='both', which='both', width=1, length=4)

ax_box.margins(y=0.01)


ax_box.set_ylim(pos[0]-gap*0.6, pos[-1]+gap*0.6)

# ----- summary table -----
ax_tbl.axis('off')
tbl = ax_tbl.table(cellText=summary.values,
                   rowLabels=summary.index,
                   colLabels=summary.columns,
                   cellLoc='center', rowLoc='center',
                   loc='center')
                   
tbl.auto_set_font_size(False) 
tbl.set_fontsize(22)     
tbl.scale(1, 1.5)
ax_tbl.set_title("Summary Statistics for all Models", fontsize=26)
plt.show()

