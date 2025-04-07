import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Read CSV
results = pd.read_csv("results.csv")

# Define sums and counts for each size category
small_sum = 0
small_count = 0

medium_sum = 0
medium_count = 0

large_sum = 0
large_count = 0

# Loop through each row to accumulate sums
for i in range(len(results)):
    model_name = results.iloc[i]["Model"]
    mse_value  = results.iloc[i]["MSE"]
    
    # Check which size category this row belongs to
    if re.search("small", model_name, re.IGNORECASE):
        small_sum += mse_value
        small_count += 1
    elif re.search("medium", model_name, re.IGNORECASE):
        medium_sum += mse_value
        medium_count += 1
    elif re.search("large", model_name, re.IGNORECASE):
        large_sum += mse_value
        large_count += 1

# Compute averages (handle the case where count=0 to avoid division by zero)
small_avg  = small_sum  / small_count  if small_count  > 0 else np.nan
medium_avg = medium_sum / medium_count if medium_count > 0 else np.nan
large_avg  = large_sum  / large_count  if large_count  > 0 else np.nan

# Plot settings
plt.rcParams.update({
    'font.size': 16,        # General font size
    'axes.titlesize': 18,   # Title font
    'axes.labelsize': 16,   # Axis labels
    'xtick.labelsize': 14,  # X tick labels
    'ytick.labelsize': 14   # Y tick labels
})

# Make the bar chart
values   = [small_avg, medium_avg, large_avg]
category = ["Small", "Medium", "Large"]

plt.figure(figsize=(12, 6))
bars = plt.bar(category, values, width=0.8)  # keep the returned bar objects
plt.title("Relationship of Size and MSE")
plt.ylabel("MSE")
plt.xlabel("Size")

# Print the computed averages
print("Small average MSE: ", small_avg)
print("Medium average MSE:", medium_avg)
print("Large average MSE: ", large_avg)

# Annotate the bars with their numerical values
for bar in bars:
    yval = bar.get_height()
    # Format to 2 decimal places (or change as desired)
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval,
        f"{yval:.2f}",
        ha="center",
        va="bottom"
    )

plt.show()
