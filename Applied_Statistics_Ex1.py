import numpy as np
import pandas as pd
from collections import Counter

data = pd.read_csv('E:\Subash\AI\Stats.csv')

# Means - Sum of elements / Number of elements
print 'Means  - Salary of Employees: ', np.mean(data['Salary in Rs.'])

# Median - It the middle Number
print 'Median  - Salary of Employee: ', np.median(data['Salary in Rs.'])

# Mode - Which occurs most frequently
print 'Mode - Salary of Employee: ', Counter(data['Salary in Rs.']).most_common(1)