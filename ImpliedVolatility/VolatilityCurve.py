import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from Data.Tradier import get_tradier_data 

def get_volatility_curve():
    """
    Fits Gaussian Process Regression to Implied Volatility
    """

    X, y, strikes = get_tradier_data()

    rng = np.random.RandomState(1)
    training_indices = rng.choice(np.arange(y.size), size = len(y), replace = False)
    X_train, y_train = X[training_indices], y[training_indices]

    kernel = RBF(length_scale = 1.0, length_scale_bounds = (1e-2, 1e2))
    gaussian_process = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 10)

    gaussian_process.fit(X_train, y_train)
    mean_prediction1, std_prediction1 = gaussian_process.predict(X, return_std = True)

    noise_std = 0.02
    y_train_noisy = y_train + rng.normal(loc = 0.0, scale = noise_std, size = y_train.shape)

    gaussian_process = GaussianProcessRegressor(kernel = kernel, alpha = noise_std**2, n_restarts_optimizer = 10)
    gaussian_process.fit(X_train, y_train_noisy)
    mean_prediction2, std_prediction2 = gaussian_process.predict(X, return_std=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle("Gaussian Process Regression")

    ax1.plot(strikes, y, linestyle = "dotted", label = "Dataset")
    ax1.plot(strikes, mean_prediction1, linestyle = "dashdot", label = "Gaussian Prediction")
    ax1.axvline(x = 396.03, linestyle = ":")
    ax1.set_title("Ridge Regression")
    ax1.legend(loc = "best")

    ax2.plot(strikes, y, linestyle = "dotted", label = "Dataset")
    ax2.plot(strikes, mean_prediction2, linestyle = "dashdot", label = "Gaussian Prediction")
    ax2.axvline(x = 396.03, linestyle = ":")
    ax2.set_title("Ridge Regression with Noise")
    ax2.legend(loc = "best")

    ax3.plot(strikes, (y - mean_prediction1), linestyle = "dotted", label = "Market - Model")
    ax3.set_title("Ridge Regression Residual")
    ax3.legend(loc = "best")

    ax4.plot(strikes, (y - mean_prediction2), linestyle = "dotted", label = "Market - Model")
    ax4.set_title("Ridge Regression with Noise Residual")
    ax4.legend(loc = "best")

    plt.show()

    return 0 