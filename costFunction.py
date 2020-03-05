import numpy as numpy
import pandas as panda
import matplotlib.pyplot as plt

# proccessing input data
traning_data = panda.read_csv('input.csv')
plt.rcParams['figure.figsize'] = (12.0, 9.0)
# access data from 1st column- index of 1st column is 0
x = traning_data.iloc[:, 0]
# access data from 2nd column -index of 2nd column is 1
y = traning_data.iloc[:, 1]
# plt.scatter(x, y)
# plt.show()
# ------------------------------------------------------------------------------------
# building a model
# initially
m = 0
c = 0

learning_rate = 0.0001
no_of_iteration = 1000

training_set_count =len(x)

# start gradient descent process
# predict the value of y with current value m and the partial derivatives
for i in range(training_set_count):
    y_predict = m * x + c

    D_m = (-2 / training_set_count) * (sum(x * (y - y_predict)))
    D_c = (-2 / training_set_count) * (sum(y - y_predict))

    # then update the value of m and c
    m = m - learning_rate * D_m
    c = c - learning_rate * D_c
print("optimum value of m", m)
print("optimum value of c ", c)

#-------------------------------------------
# making a prediction is the
y_predict = m * x + c
plt.scatter(x,y)
plt.plot([min(x),max(y)],[min(y_predict),max(y_predict)],color='red')
plt.show()
