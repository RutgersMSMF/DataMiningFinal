import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression

def example_fit():
    """
    SkLearn Example Tutorial
    """

    X = np.linspace(0, 10, 1_000).reshape(-1, 1)
    y = X * np.sin(X)

    rng = np.random.RandomState(1)
    training_indices = rng.choice(np.arange(y.size), size = 6, replace = False)
    X_train, y_train = X[training_indices], y[training_indices]

    kernel = RBF(length_scale = 1.0, length_scale_bounds = (1e-2, 1e2))
    gaussian_process = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 9)

    gaussian_process.fit(X_train, y_train)
    mean_prediction1, std_prediction1 = gaussian_process.predict(X, return_std = True)

    noise_std = 0.75
    y_train_noisy = y_train + rng.normal(loc=0.0, scale=noise_std, size=y_train.shape)

    gaussian_process = GaussianProcessRegressor(kernel = kernel, alpha = noise_std**2, n_restarts_optimizer = 9)
    gaussian_process.fit(X_train, y_train_noisy)
    mean_prediction2, std_prediction2 = gaussian_process.predict(X, return_std=True)

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)
    linear_prediction = linear_regression.predict(X)

    linear_regression_noise = LinearRegression()
    linear_regression_noise.fit(X_train, y_train_noisy)
    linear_prediction_noise = linear_regression.predict(X)

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle("Gaussian Process Regression")

    ax1.plot(y, linestyle = "dotted", label = "Dataset")
    ax1.plot(mean_prediction1, linestyle = "dashdot", label = "Gaussian Prediction")
    ax1.plot(linear_prediction, linestyle = "dashed", label = "Linear Prediction")
    ax1.set_title("Without Noise")
    ax1.legend(loc = "best")

    ax2.plot(y, linestyle = "dotted", label = "Dataset")
    ax2.plot(mean_prediction2, linestyle = "dashdot", label = "Gaussian Prediction")
    ax2.plot(linear_prediction_noise, linestyle = "dashed", label = "Linear Prediction")
    ax2.set_title("With Noise")
    ax2.legend(loc = "best")

    plt.show()

    return 0



