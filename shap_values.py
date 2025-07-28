#!/usr/bin/env python3
import torch
import shap
import numpy as np
import pandas as pd
from create_model import create_model

def main():
    # 1. Device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(4, [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088], 1).to(device)
    state_dict = torch.load("models/large_layers_left_pyramid_250_5e-05_Adam.pt", map_location=device)

    new_state_dict = {
        k.replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(new_state_dict)
    model.eval()

    # 2. Load data and make tensors
    X_train = pd.read_csv("data/X_train.csv")
    X_test  = pd.read_csv("data/X_test.csv")
    X_train_t = torch.FloatTensor(X_train.values).to(device)
    X_test_t  = torch.FloatTensor(X_test.values[:10]).to(device)

    # 3. Build explainer using *all* train points
    explainer = shap.GradientExplainer(model, X_train_t)

    # 4. Compute SHAP values
    sv = explainer.shap_values(X_test_t)
    if isinstance(sv, list):
        sv = sv[0]
    sv = sv.squeeze(-1)
    if torch.is_tensor(sv):
        sv = sv.detach().cpu().numpy()

    # 5. Mean absolute SHAP per feature
    feature_names = X_test.columns.tolist()
    mean_abs = np.abs(sv).mean(axis=0)

    # 6. Print in descending order
    sorted_idx = np.argsort(mean_abs)[::-1]
    print("Feature\tMean |SHAP|")
    for i in sorted_idx:
        print(f"{feature_names[i]:<30s} {mean_abs[i]:.5f}")


if __name__ == "__main__":
    main()
