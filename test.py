import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

results = pd.read_csv("results.csv")
Models = results["Model"]
MSE = results["MSE"]
R2 =  results["R^2"]
MAE =  results["MAE"]
CV =  results["CV"]
Count =  results["Count"]



#NUMBER OF EPOCH ANALYSIS
categories = ["100 Epochs", "250 Epochs", "500 Epochs"]
values = [0,0,0]
count = 0

for i in range(len(results)):
    if re.search("100", results.iloc[i]["Model"]):
        values[0] += results.iloc[i]["MSE"]
        count += 1
    elif re.search("250", results.iloc[i]["Model"]):
        values[1] += results.iloc[i]["MSE"]
    elif re.search("500", results.iloc[i]["Model"]):
        values[2] += results.iloc[i]["MSE"]
    else:
        print("ERROR no epoch found in string")

for i in range(3):
    values[i] /= count
    print(values[i])
    
plt.bar(categories, values)
for index, value in enumerate(values):
    plt.text(index, value, f"{value:.5f}", ha="center", va="bottom")
plt.xlabel("# of Epochs")
plt.ylabel("Average Mean Standard Error")
plt.title("Relationship of Epochs on Error")
plt.show()





#Effect of Size and Epoch
categories = ["100 Epochs Small", "100 Epochs Medium", "100 Epochs Large", "250 Epochs Small", "250 Epochs Medium", "250 Epochs Large", "500 Epochs Small", "500 Epochs Medium", "500 Epochs Large"]
values = [0,0,0,0,0,0,0,0,0]
count = 0

for i in range(len(results)):
    if re.search("100", results.iloc[i]["Model"]) and re.search("small", results.iloc[i]["Model"]):
        values[0] += results.iloc[i]["MSE"]
        count += 1
    elif re.search("100", results.iloc[i]["Model"]) and re.search("medium", results.iloc[i]["Model"]):
        values[1] += results.iloc[i]["MSE"]
    elif re.search("100", results.iloc[i]["Model"]) and re.search("large", results.iloc[i]["Model"]):
        values[2] += results.iloc[i]["MSE"]
        
    elif re.search("250", results.iloc[i]["Model"]) and re.search("small", results.iloc[i]["Model"]):
        values[3] += results.iloc[i]["MSE"]
    elif re.search("250", results.iloc[i]["Model"]) and re.search("medium", results.iloc[i]["Model"]):
        values[4] += results.iloc[i]["MSE"]
    elif re.search("250", results.iloc[i]["Model"]) and re.search("large", results.iloc[i]["Model"]):
        values[5] += results.iloc[i]["MSE"]
        
    elif re.search("500", results.iloc[i]["Model"]) and re.search("small", results.iloc[i]["Model"]):
        values[6] += results.iloc[i]["MSE"]
    elif re.search("500", results.iloc[i]["Model"]) and re.search("medium", results.iloc[i]["Model"]):
        values[7] += results.iloc[i]["MSE"]
    elif re.search("500", results.iloc[i]["Model"]) and re.search("large", results.iloc[i]["Model"]):
        values[8] += results.iloc[i]["MSE"]
        
        
for i in range(9):
    values[i] /= count
    print(values[i])
    
plt.bar(categories, values)
for index, value in enumerate(values):
    plt.text(index, value, f"{value:.5f}", ha="center", va="bottom")
plt.xlabel("# of Epochs")
plt.ylabel("Average Mean Standard Error")
plt.title("Relationship of Epochs and Size on Error")
plt.show()










#Effect of Shape
categories = ["Left Pyramid", "Right Pyramid", "Diamond", "Block"]
values = [0 for i in range(len(categories))]
count = 0

for i in range(len(results)):
    if re.search("left", results.iloc[i]["Model"]):
        values[0] += results.iloc[i]["MSE"]
        count += 1
    elif re.search("right", results.iloc[i]["Model"]):
        values[1] += results.iloc[i]["MSE"]
    elif re.search("diamond", results.iloc[i]["Model"]):
        values[2] += results.iloc[i]["MSE"]
    elif re.search("block", results.iloc[i]["Model"]):
        values[3] += results.iloc[i]["MSE"]
    else:
        print("ERROR no epoch found in string")

for i in range(len(categories)):
    values[i] /= count
    print(values[i])
    
plt.bar(categories, values)
for index, value in enumerate(values):
    plt.text(index, value, f"{value:.5f}", ha="center", va="bottom")
plt.xlabel("Shape")
plt.ylabel("Average Mean Standard Error")
plt.title("Relationship of Shape on Error")
plt.show()




