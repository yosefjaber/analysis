import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Read CSV
results = pd.read_csv("results.csv")

# Define sums and counts for each size category
small_small = 10000
big_small = -10000

small_medium = 10000
big_medium = -10000

small_large = 10000
big_large = -10000

# Loop through each row to accumulate sums
for i in range(len(results)):
    model_name = results.iloc[i]["Model"]
    mse_value  = results.iloc[i]["MSE"]
    
    # Check which size category this row belongs to
    if re.search("small", model_name, re.IGNORECASE):
        if small_small > mse_value:
            small_small = mse_value
        elif big_small < mse_value:
            big_small = mse_value
                
    elif re.search("medium", model_name, re.IGNORECASE):
        if small_medium > mse_value:
            small_medium = mse_value
        elif big_medium < mse_value:
            big_medium = mse_value
    elif re.search("large", model_name, re.IGNORECASE):
        if small_large > mse_value:
            small_large = mse_value
        elif big_large < mse_value:
            big_large = mse_value

# Compute averages (handle the case where count=0 to avoid division by zero)
small_diff = big_small - small_small
medium_diff = big_medium - small_medium
large_diff = big_large - small_large

# Plot settings
plt.rcParams.update({
    'font.size': 16,        # General font size
    'axes.titlesize': 18,   # Title font
    'axes.labelsize': 16,   # Axis labels
    'xtick.labelsize': 14,  # X tick labels
    'ytick.labelsize': 14   # Y tick labels
})

# Make the bar chart
values   = [small_diff, medium_diff, large_diff]
category = ["Small Difference", "Medium Difference", "Large Difference"]

plt.figure(figsize=(12, 6))
bars = plt.bar(category, values, width=0.8)  # keep the returned bar objects
plt.title("Difference between highest and lowest value for each size")
plt.ylabel("MSE")
plt.xlabel("Size")

# Print the computed averages
print("Small average MSE: ", small_diff)
print("Medium average MSE:", medium_diff)
print("Large average MSE: ", large_diff)

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
