# for array computations and loading data
import numpy as np

# for building linear regression models and preparing data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def train_val_test_split(x, y, train_size, test_val_prop):
    test_val_size = 1 - train_size
    # Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
    x_train, x_, y_train, y_ = train_test_split(x, y, test_size=test_val_size, random_state=1)

    # Split the 40% subset above into two: one half for cross validation and the other for the test set
    x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=test_val_prop, random_state=1)

    # Delete temporary variables
    del x_, y_

    print(f"the shape of the training set (input) is: {x_train.shape}")
    print(f"the shape of the training set (target) is: {y_train.shape}\n")
    print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
    print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")
    print(f"the shape of the test set (input) is: {x_test.shape}")
    print(f"the shape of the test set (target) is: {y_test.shape}")

    return x_train, y_train, x_cv, y_cv, x_test, y_test


def scaling(x_train, x_cv):
    # Initialize the class
    scaler_linear = StandardScaler()

    # Compute the mean and standard deviation of the training set then transform it
    X_train_scaled = scaler_linear.fit_transform(x_train)
    print(f"Computed mean of the training set: {scaler_linear.mean_.squeeze():.2f}")
    print(f"Computed standard deviation of the training set: {scaler_linear.scale_.squeeze():.2f}")

    X_cv_scaled = scaler_linear.transform(x_cv)
    print(f"Mean used to scale the CV set: {scaler_linear.mean_.squeeze():.2f}")
    print(f"Standard deviation used to scale the CV set: {scaler_linear.scale_.squeeze():.2f}")

    return X_train_scaled, X_cv_scaled


def scaling_with_polynomial(x_train, x_cv, poly_degree):
    """ Make polynomial features & scale"""
    # Instantiate the class to make polynomial features
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    # Compute the number of features and transform the training set
    X_train_mapped = poly.fit_transform(x_train)
    # Add the polynomial features to the cross validation set
    X_cv_mapped = poly.transform(x_cv)

    X_train_mapped_scaled, X_cv_mapped_scaled = scaling(X_train_mapped, X_cv_mapped)

    return X_train_mapped_scaled, X_cv_mapped_scaled


def train_model(X_train_scaled, y_train):
    # Initialize the class
    linear_model = LinearRegression()
    # Train the model
    linear_model.fit(X_train_scaled, y_train)

    return linear_model


def predict(X_train_scaled, linear_model):
    """Feed the scaled training set and get the predictions"""
    ypred = linear_model.predict(X_train_scaled)
    return ypred


def mse_with_sklearn(y_train, ypred):
    """Use scikit-learn's utility function and divide by 2"""
    mse = mean_squared_error(y_train, ypred) / 2
    print(f"training MSE (using sklearn function): {mean_squared_error(y_train, yhat) / 2}")

    return mse


def mse_with_loop(y_train, ypred):
    """for-loop implementation"""
    total_squared_error = 0

    for i in range(len(ypred)):
        squared_error_i = (ypred[i] - y_train[i]) ** 2
        total_squared_error += squared_error_i

    mse = total_squared_error / (2 * len(ypred))
    print(f"training MSE (for-loop implementation): {mse.squeeze()}")

    return mse