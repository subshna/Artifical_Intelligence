import numpy as np
import pandas as pd
import math

data = pd.read_csv('E:\Subash\AI\Stats.csv')

# Variance - Sum of all element minus Mean value, difference is squared and added and divide by number of data points - 1
m = np.mean(data['Salary in Rs.'])
n = len(data)
Variance = (sum((x - m)**2 for x in (data['Salary in Rs.'])) / (n - 1))
print round(Variance, 2)

# Standard deviation - Square root of the variance
SD = math.sqrt(Variance)
print round(SD, 2)