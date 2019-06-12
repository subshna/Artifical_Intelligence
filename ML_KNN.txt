import pandas as pd
import numpy as np
from sklearn import model_selection, neighbors

in_filename = 'E:\Subash\AI\Census_Data.csv'
df = pd.read_csv(in_filename, sep=',', index_col=None)
df.replace('?', -99999, inplace=True)
x = np.array(df.drop(['Survival_Status'],1))
y = np.array(df['Survival_Status'])

# Split the file into train [80%] and test [20%] data
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.20)

# Apply KNN algorithm
knn = neighbors.KNeighborsClassifier()
# Train the model using the training set
knn.fit(x_train, y_train)

# Predict using test data
predict = knn.predict(x_test)
print 'Prediction using test data: ', predict

# Perform Accuracy check
accuracy = knn.score(x_test, y_test)
print 'Accuracy: ', accuracy