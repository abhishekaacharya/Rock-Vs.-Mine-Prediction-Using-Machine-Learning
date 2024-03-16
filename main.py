# importing required libraries and modules
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#storing LogisticRegression in variable called "lg"
lg = LogisticRegression()
#Converting the data into DataFrame Using Pandas
data = pd.read_csv('sonardata.csv', header=None)
# print(data.sample(20)) #printing random data

#slicing the data such that 'x' contains all the features and 'y' contains the output/result
x = data.drop(columns=60, axis=1)
y = data[60]
# print(x.sample(10)) #checking sample data of x
# print(y.sample(10)) # checking sample data of y

#Dividing the data into training data and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
#After dividing the data, fit the training data into logistic-regression mdodel
lg.fit(x_train, y_train)
#prediction of training data
pre = lg.predict(x_train)
# finding accuracy of our Model
accuracy = accuracy_score(pre, y_train)
########################################################################################
print(f'accuracy of the model is {accuracy * 100}')
########################################################################################

#prediction for real time vales
#store the feature values into input data
#Below data is take from the CSV file
input_data = (0.0216,0.0215,0.0273,0.0139,0.0357,0.0785,0.0906,0.0908,0.1151,0.0973,
              0.1203,0.1102,0.1192,0.1762,0.2390,0.2138,0.1929,0.1765,0.0746,0.1265,
              0.2005,0.1571,0.2605,0.5386,0.8440,1.0000,0.8684,0.6742,0.5537,0.4638,
              0.3609,0.2055,0.1620,0.2092,0.3100,0.2344,0.1058,0.0383,0.0528,0.1291,
              0.2241,0.1915,0.1587,0.0942,0.0840,0.0670,0.0342,0.0469,0.0357,0.0136,
              0.0082,0.0140,0.0044,0.0052,
              0.0073,0.0021,0.0047,0.0024,0.0009,0.0017)
# converting the input data into the array using numpy 'asarray'
input_array = np.asarray(input_data)
#reshape the obtained array using 'reshape'
reshaped_input = input_array.reshape(1, -1)
#prediction of the real time values
prediction = lg.predict(reshaped_input)
if prediction == "R":
    print(f'The Object is Rock')
elif prediction == "M":
    print(f'The object is Mine')    