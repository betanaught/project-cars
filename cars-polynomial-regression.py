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

# Split into testing and training sets
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test  = cdf[~msk]
train_x = np.asanyarray(test[['ENGINESIZE']])
train_y = np.asanyarray(test[['CO2EMISSIONS']])
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

# ------------------------- QUADRADIC REGRESSION ------------------------
quad = PolynomialFeatures(degree = 2)
train_x_quad = quad.fit_transform(train_x)
train_x_quad

clf_quad = linear_model.LinearRegression()
train_y_quad = clf_quad.fit(train_x_quad, train_y)
# Print coefficients for quadratic fit
print('Coefficients: ', clf_quad.coef_)
print('Intercept: ', clf_quad.intercept_)

# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
# XX = np.arange(0.0, 10.0, 0.1)
# yy = clf.intercept_[0]+clf.coef_[0][1]*XX+clf.coef_[0][2]*np.power(XX, 2)
# plt.plot(XX, yy, 'r')
# plt.xlabel("Engine Size")
# plt.ylabel("Emissions")

from sklearn.metrics import r2_score

test_x_quad = quad.fit_transform(test_x)
test_y_quad = clf_quad.predict(test_x_quad)

print("Mean absolute error: %.2f" %
      np.mean(np.absolute(test_y_quad - test_y)))
print("Residual sum of squares (MSE): %.2f" %
      np.mean((test_y_quad - test_y)**2))
print("R2-score: %.2f" % r2_score(test_y_quad, test_y))

# --------------------------- CUBIC REGRESSION ----------------------------
cubic = PolynomialFeatures(degree = 3)
train_x_cubic = cubic.fit_transform(train_x)

clf_cubic = linear_model.LinearRegression()
train_y_cubic = clf_cubic.fit(train_x_cubic, train_y)
print('Coefficients: ', clf_cubic.coef_)
print('Intercept: ', clf_cubic.intercept_)

test_x_cubic = cubic.fit_transform(test_x)
test_y_cubic = clf_cubic.predict(test_x_cubic)

print("Mean absolute error: %.2f" %
      np.mean(np.absolute(test_y_cubic - test_y)))
print("Residual sum of squares (MSE): %.2f" %
      np.mean((test_y_cubic - test_y)**2))
print("R2-score: %.2f" % r2_score(test_y_cubic, test_y))

# -------------------------- CUBIC SOLUTION ------------------------------
poly3 = PolynomialFeatures(degree=3)
train_x_poly3 = poly3.fit_transform(train_x)
clf3 = linear_model.LinearRegression()
train_y3_ = clf3.fit(train_x_poly3, train_y)

print('Coefficients: ', clf3.coef_)
print('Intercept: ', clf3.intercept_)

test_x_poly3 = poly3.fit_transform(test_x)
test_y3_ = clf3.predict(test_x_poly3)
print("Mean absolute error: %.2f" %
      np.mean(np.absolute(test_y3_ - test_y)))
print("MSE: %.2f" %
      np.mean((test_y3_ - test_y)**2))
print("R2-Score: %.2f" % r2_score(test_y3_ , test_y))

