import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import  mean_squared_error
brc = datasets.load_breast_cancer()
brc.keys()
brc_X = brc.data[:, np.newaxis, 2]

brc_X_train = brc_X[:-30]
brc_X_test =brc_X[-30:]

brc_y_train = brc.target[:-30]
brc_y_test =brc.target[-30:]

# model = linear_model.LinearRegression()
model = linear_model.LinearRegression()

model.fit(brc_X_train, brc_y_train)
brc_y_predicted = model.predict(brc_X_test)

print("mean squared error is :", mean_squared_error(brc_y_test, brc_y_predicted))

print("weights:" , model.coef_)
print("intercept:" , model.intercept_)

plt.scatter(brc_X_test, brc_y_test)
plt.plot(brc_X_test, brc_y_predicted)

plt.show()
