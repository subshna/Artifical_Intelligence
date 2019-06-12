import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn import linear_model
from sklearn.model_selection import train_test_split

in_filename = 'E:\Subash\AI\Car_MPG_Data.csv'
df = pd.read_csv(in_filename, sep=',', index_col=None)

# Load the Data
x = pd.DataFrame(df, columns=['Acceleration'])
y = pd.DataFrame(df, columns=['MPG'])

# Train [80%] and test [20%] data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Apply linear regression
reg = linear_model.LinearRegression()
lm = reg.fit(x_train, y_train)

# Display the coefficients coef, intercept and residues
print 'Coefficeint: ', reg.coef_

# Predict using test data
predict_outcome = reg.predict(x_test)
print predict_outcome[0:5]

# Perform Accuracy check using the R Square
print 'Mean Squared Error: %.2f' % np.mean((predict_outcome - y_test)**2)
print 'Variance Score: %.2f' % reg.score(x_test, y_test)

# Display using scatter plot the data points and the best fit line
plot.scatter(x_test, y_test, color='black')
plot.plot(x_test, predict_outcome, color='blue')
plot.xlabel('True')
plot.ylabel('Prediction')
plot.show()