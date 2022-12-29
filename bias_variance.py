""""
- Underfit: High Bias   --  J_train is High  --  J_cv is high
- Just Right: --  J_train is Low  --  J_cv is Low
- Overfit: High Variance  --  J_train is Low  --  J_cv is High

To fix a HIGH BIAS problem, you can:
- try adding polynomial features
- try getting additional features
- try decreasing the regularization parameter
- NO MATTER HOW MANY DATA YOU ADD, IT DOESN'T GIVE BETTER TRAINING ERROR

To fix a HIGH VARIANCE problem, you can:
- try increasing the regularization parameter
- try smaller sets of features
- get more training examples

How to decide low or high?
- Baseline performance --- training error --- cv error
- If training error is high compare to baseline error: high bias - underfit (decrease reg, add poly, get addtional feature)
- Then, check the performance on validation:
- If valid error is high compare to training error: high variance- overfit (increase reg, get more data, try smaller feature)

NEURAL NETWORK CYCLE
- is the error well on training set ->(N - high bias - underfit!) it means high bias, get bigger network  --> check training error
- is the error well on training set ->(Y) is it well on cv -->(N - high variance - overfit!) --> more data --> check training error
- is it well on cv --> (Y) --> OK!
"""
import numpy as np
# for building linear regression models and preparing data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge


##################
# PLOT DIFFERENT POLY AND REGULARIZED MODELS
##################
def plot_params_mse(params, train_mses, cv_mses, baseline):
    # Plot the results
    plt.plot(params, train_mses, marker='o', c='r', label='training MSEs');
    plt.plot(params, cv_mses, marker='o', c='b', label='CV MSEs');
    plt.plot(params, np.repeat(baseline, len(params)), linestyle='--', label='baseline')
    plt.title(str(params) + " vs. train and CV MSEs")
    plt.xticks(degrees)
    plt.xlabel(str(params))
    plt.ylabel("MSE");
    plt.legend()
    plt.show()


##################
# LOAD DATA
##################
data = np.loadtxt('data/c2w3_lab2_data1.csv', delimiter=",")

x = data[:, :-1]
y = data[:, -1]

# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables.
x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=80)

# Split the 40% subset above into two: one half for cross validation and the other for the test set
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=80)

print(f"the shape of the training set (input) is: {x_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")


##################
# TRY DIFFERENT POLY MODELS
##################
max_degree = 10
baseline = 400
train_mses = []
cv_mses = []
models = []
scalers = []
degrees = range(1, max_degree+1)

reg_params_list = [0.5, 1, 1.5]
reg_params = [str(x) for x in reg_params_list]
model = LinearRegression()

# Loop over 10 times. Each adding one more degree of polynomial higher than the last.
for degree in degrees:
    # Add polynomial features to the training set
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_mapped = poly.fit_transform(x_train)

    # Scale the training set
    scaler_poly = StandardScaler()
    X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
    scalers.append(scaler_poly)

    # Create and train the model
    model.fit(X_train_mapped_scaled, y_train )
    models.append(model)

    # Compute the training MSE
    yhat = model.predict(X_train_mapped_scaled)
    train_mse = mean_squared_error(y_train, yhat) / 2
    train_mses.append(train_mse)

    # Add polynomial features and scale the cross-validation set
    poly = PolynomialFeatures(degree, include_bias=False)
    X_cv_mapped = poly.fit_transform(x_cv)
    X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

    # Compute the cross-validation MSE
    yhat = model.predict(X_cv_mapped_scaled)
    cv_mse = mean_squared_error(y_cv, yhat) / 2
    cv_mses.append(cv_mse)

plot_params_mse(degrees, train_mses, cv_mses, baseline)

optimal_degree = np.argmin(cv_mses) + 1


##################
# TRY DIFFERENT REGULARIZED MODELS
##################
baseline = 400
train_mses = []
cv_mses = []
models = []
scalers = []
reg_params_list = [0.5, 1, 1.5]
reg_params = [str(x) for x in reg_params_list]

# Loop over 10 times. Each adding one more degree of polynomial higher than the last.
for reg_param in reg_params:
    # Add polynomial features to the training set
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_mapped = poly.fit_transform(x_train)

    # Scale the training set
    scaler_poly = StandardScaler()
    X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
    scalers.append(scaler_poly)

    # Create and train the model
    model = Ridge(alpha=reg_param)
    model.fit(X_train_mapped_scaled, y_train)
    models.append(model)

    # Compute the training MSE
    yhat = model.predict(X_train_mapped_scaled)
    train_mse = mean_squared_error(y_train, yhat) / 2
    train_mses.append(train_mse)

    # Add polynomial features and scale the cross-validation set
    poly = PolynomialFeatures(degree, include_bias=False)
    X_cv_mapped = poly.fit_transform(x_cv)
    X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

    # Compute the cross-validation MSE
    yhat = model.predict(X_cv_mapped_scaled)
    cv_mse = mean_squared_error(y_cv, yhat) / 2
    cv_mses.append(cv_mse)

plot_params_mse(reg_params, train_mses, cv_mses, baseline)


##################
# LOAD DATA FOR MORE TRAINING EXAMPLES
##################
data = np.loadtxt('data/c2w3_lab2_data4.csv', delimiter=",")

x = data[:, :-1]
y = data[:, -1]

# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables.
x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=80)

# Split the 40% subset above into two: one half for cross validation and the other for the test set
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=80)

print(f"the shape of the training set (input) is: {x_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")


##################
# GET MORE TRAINING DATA
##################
train_mses = []
cv_mses = []
models = []
scalers = []
num_samples_train_and_cv = []
degree = 4
baseline = 250
percents = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

model = LinearRegression()

# Loop over 10 times. Each adding one more degree of polynomial higher than the last.
for percent in percents:
    num_samples_train = round(len(x_train) * (percent / 100.0))
    num_samples_cv = round(len(x_cv) * (percent / 100.0))
    num_samples_train_and_cv.append(num_samples_train + num_samples_cv)

    x_train_sub = x_train[:num_samples_train]
    y_train_sub = y_train[:num_samples_train]
    x_cv_sub = x_cv[:num_samples_cv]
    y_cv_sub = y_cv[:num_samples_cv]

    # Add polynomial features to the training set
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_mapped = poly.fit_transform(x_train_sub)

    # Scale the training set
    scaler_poly = StandardScaler()
    X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
    scalers.append(scaler_poly)

    # Create and train the model
    model.fit(X_train_mapped_scaled, y_train_sub)
    models.append(model)

    # Compute the training MSE
    yhat = model.predict(X_train_mapped_scaled)
    train_mse = mean_squared_error(y_train_sub, yhat) / 2
    train_mses.append(train_mse)

    # Add polynomial features and scale the cross-validation set
    poly = PolynomialFeatures(degree, include_bias=False)
    X_cv_mapped = poly.fit_transform(x_cv_sub)
    X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

    # Compute the cross-validation MSE
    yhat = model.predict(X_cv_mapped_scaled)
    cv_mse = mean_squared_error(y_cv_sub, yhat) / 2
    cv_mses.append(cv_mse)

plot_params_mse(num_samples_train_and_cv, train_mses, cv_mses, baseline)

