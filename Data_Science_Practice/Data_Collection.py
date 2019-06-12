# Import library to read data into dataframe
import pandas as pd
# Import numpy library
import numpy as np
# Import library from regular expression
import re

pd.set_option('display.max_columns', None)
datacontent = pd.read_csv('E:\Subash\Python\Data_Science_Practice\NSE-WIPRO.csv')
print ('Reading data to Dataframe')

print (datacontent.head())
print (datacontent.shape)

match.group(0) for