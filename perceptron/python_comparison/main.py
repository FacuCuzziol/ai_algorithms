from sklearn.linear_model import Perceptron
import pandas as pd
import matplotlib.pyplot as plt
# import simple_dataset.csv file
data = pd.read_csv('simple_dataset.csv')
# set column names
data.columns = ['x1', 'x2', 'x3', 'y']
print(data)
# graph the data, red for y=-1, blue for y=1, it will run in non gui backend
plt.scatter(data['x2'], data['x3'], c=data['y'])
plt.show()
# create a perceptron object with 100 iterations
perceptron = Perceptron(max_iter=100)
# simple_dataset has tree columns of train data and one of target data, at the end
# we need to separate the target data from the train data
pivot = 0.45
# shuffle the data
data = data.sample(frac=1)
# separate the train data from the target data
train_data = data[:int(len(data)*pivot)]
target_data = data[int(len(data)*pivot):]
# train the perceptron
perceptron.fit(train_data[['x1', 'x2', 'x3']], train_data['y'])
# predict
predictions = perceptron.predict(target_data[['x1', 'x2', 'x3']])
# print the predictions
print(predictions)
# print the data
print(target_data)
# print the accuracy
print(perceptron.score(target_data[['x1', 'x2', 'x3']], target_data['y']))
