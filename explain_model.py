import torch
import shap
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from create_model import create_model
import matplotlib.pyplot as plt  

plt.rcParams.update({
    "font.size": 200,          # base size for everything
    "axes.titlesize": 28,     # title of the plot
    "axes.labelsize": 20,     # axis labels
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_model(4, [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088], 1).to(device)

state_dict = torch.load("models/large_layers_left_pyramid_250_5e-05_Adam.pt", map_location=device)

new_state_dict = {
    k.replace("_orig_mod.", ""): v
    for k, v in state_dict.items()
}
model.load_state_dict(new_state_dict)
model.eval()

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")
y_test = pd.read_csv("data/y_test.csv")

X_train = torch.FloatTensor(X_train.values)
X_test = torch.FloatTensor(X_test.values)
y_train = torch.FloatTensor(y_train.values)
y_test = torch.FloatTensor(y_test.values)

bg_idx     = np.random.choice(len(X_train), size=1000, replace=False)
background = X_train[bg_idx].to(device)
explainer  = shap.GradientExplainer(model, background)
test_loader = DataLoader(X_test, batch_size=32, shuffle=False)
inputs      = X_test.to(device)

sv = explainer.shap_values(inputs)          # (32, 4, 1)  *or* list[...]

# if your SHAP version returns a list, keep the first element
if isinstance(sv, list):
    sv = sv[0]                              # still (32, 4, 1)

# remove the trailing length‑1 axis
sv = sv.squeeze(-1)                         # (32, 4)

# make sure it’s a NumPy array on CPU
if torch.is_tensor(sv):
    sv = sv.detach().cpu().numpy()

feature_names = pd.read_csv("data/X_test.csv").columns.tolist()
print(feature_names)

n_feats = len(feature_names)
shap.summary_plot(
    sv,
    inputs.cpu().numpy(),
    feature_names=feature_names,
    max_display=n_feats,
    show = False
)

fig = plt.gcf()

# loop through all axes (main plot + colorbar)
for ax in fig.axes:
    # set title / axis labels
    ax.title.set_fontsize(26)
    ax.xaxis.label.set_fontsize(26)
    ax.yaxis.label.set_fontsize(26)
    # set tick labels
    ax.tick_params(axis='both', which='major', labelsize=28)

plt.tight_layout()
plt.show()
