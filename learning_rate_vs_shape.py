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

#Size vs Shape
b_small = 0
_100_medium = 0
_100_large = 0

_250_small = 0
_250_medium = 0
_250_large = 0

_500_small = 0
_500_medium = 0
_500_large = 0

count = 0
for i in range(len(results)):
    model_name = results.iloc[i]["Model"]
    mse_value = results.iloc[i]["MSE"]

    if re.search("100", model_name):
        epoch_100 += mse_value
        count += 1
    elif re.search("250", model_name):
        epoch_250 += mse_value
    elif re.search("500", model_name):
       epoch_500 += mse_value
      
