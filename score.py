import pandas as pd

# Read CSV
results = pd.read_csv("results.csv")

score_rank = []

for i in range(len(results)):
    model_name = results.iloc[i]["Model"]
    mse_value  = results.iloc[i]["MSE"]
    r_value    = results.iloc[i]["R^2"]
    score = mse_value + (1 - r_value)
    score_rank.append({"Model": model_name, "Score": score})

# Sort by score (ascending: lower is better)
sorted_scores = sorted(score_rank, key=lambda x: x["Score"])

position = 1
# Print sorted results
for entry in sorted_scores:
    print(f"Model: {entry['Model']}, Score: {entry['Score']}, Rank: {position}")
    position += 1
