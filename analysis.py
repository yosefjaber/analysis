import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9, #Nothing
    "axes.labelsize": 8.6, #Bottom MSE text
    "xtick.labelsize": 8, #x tick marks
    "ytick.labelsize": 8, #y tick marks
    "axes.linewidth": 1, #Rectangle Border
    "figure.dpi": 300
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
    figsize=(8, 4.5),                    # shorter figure now
    gridspec_kw={
        'height_ratios': [1, 1],         # box‑plot axis 5 × taller
        'hspace': 0.5                    # no GridSpec gap
    }
)

# ----- boxplot -----
ax_box.boxplot(
    results['MSE'],
    vert=False,
    tick_labels=[""],
    boxprops     =dict(linewidth=1),
    whiskerprops =dict(linewidth=1),
    capprops     =dict(linewidth=1),
    medianprops  =dict(linewidth=1),
    flierprops   =dict(marker='o', markersize=2,
                       markerfacecolor='none',
                       markeredgecolor='black',
                       markeredgewidth=1)
)
ax_box.set_xlabel('MSE')
ax_box.tick_params(axis='both', which='both', width=1, length=4)

ax_box.margins(y=0.01) 
ax_box.set_yticks([]) 

ax_box.set_ylim(pos[0]-gap*0.6, pos[-1]+gap*0.6)

# ----- summary table -----
ax_tbl.axis('off')                        # hide the axis frame
tbl = ax_tbl.table(
    cellText  = summary.values,
    rowLabels = summary.index,
    colLabels = summary.columns,
    cellLoc='center', rowLoc='center',
    loc='upper center'                    # stick table to the top
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(7)

# NO tight_layout() – it would re‑insert spacing
fig.subplots_adjust(hspace=0)             # safety – keep axes touching

plt.show()

