import pandas as panda
import matplotlib.pyplot as plt
import numpy as numpy


def gradient_descent(X, Y):
    c_current_value = 0,
    m_current_value = 0
    iterration = 10000
    learning_rate = 0.08

    length_of_training_set = (len(X))
    for i in range(iterration):
        predicted_value = m_current_value * X + c_current_value
        cost = (1 / length_of_training_set) * sum([val ** 2 for val in (y - predicted_value)])

        m_d = (-2 / length_of_training_set) * sum(X * (Y - predicted_value))
        c_d = (-2 / length_of_training_set) * sum((Y - predicted_value))

        m_current_value = m_current_value - (learning_rate * m_d)
        c_current_value = c_current_value - (learning_rate * c_d)
        # each iteration print there value
        print("m {},b {},cost {}, iteration {}".format(m_current_value, c_current_value, cost, i))
        # plt.plot([min(x), max(y)], [min(predicted_value), max(predicted_value)], color='red')


x = numpy.array([1, 2, 3, 4, 5])
y = numpy.array([5, 7, 9, 11, 13])
plt.scatter(x, y)

gradient_descent(x, y)
plt.show()
