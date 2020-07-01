import matplotlib
import matplotlib.pyplot as plt
# matplotlib.interactive(False)
# matplotlib.use('PS')
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

df = pd.read_csv("FuelConsumptionCo2.csv")

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB',
          'CO2EMISSIONS']]

# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
# plt.xlabel("Engine Size")
# plt.ylabel("Emissions")
# plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test  = cdf[~msk]
train_x = np.asanyarray(test[['ENGINESIZE']])
train_y = np.asanyarray(test[['CO2EMISSIONS']])
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree = 2)
train_x_poly = poly.fit_transform(train_x)
train_x_poly

clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)
# Print coefficients for quadratic fit
print('Coefficients: ', clf.coef_)
print('Intercept: ', clf.intercept_)

# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
# XX = np.arange(0.0, 10.0, 0.1)
# yy = clf.intercept_[0]+clf.coef_[0][1]*XX+clf.coef_[0][2]*np.power(XX, 2)
# plt.plot(XX, yy, 'r')
# plt.xlabel("Engine Size")
# plt.ylabel("Emissions")

from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(test_x)
test_y_     = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" %
      np.mean((test_y_ - test_y)**2))
print("R2-score: %.2f" % r2_score(test_y_, test_y))
