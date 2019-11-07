import spafe
import pytest
import warnings
import numpy as np
import matplotlib.pyplot as plt
from spafe.utils.filters import kalman, kalman_xy

DEBUG_MODE = False
warnings.filterwarnings("ignore")

def test_functions_availability():
    # Cheching the availibility of functions in the chosen attribute
    assert hasattr(spafe.utils.filters, 'gaussian_filter')
    assert hasattr(spafe.utils.filters, 'sobel_filter')
    assert hasattr(spafe.utils.filters, 'rasta_filter')
    assert hasattr(spafe.utils.filters, 'kalman_xy')


@pytest.mark.test_id(401)
def test_kalman_xy():
    x = np.matrix('0. 0. 0. 0.').T
    P = np.matrix(np.eye(4)) * 1000  # initial uncertainty

    N = 20
    true_x = np.linspace(0.0, 10.0, N)
    true_y = true_x**2
    observed_x = true_x + 0.05 * np.random.random(N) * true_x
    observed_y = true_y + 0.05 * np.random.random(N) * true_y
    plt.plot(observed_x, observed_y, 'ro')
    result = []
    R = 0.01**2
    for meas in zip(observed_x, observed_y):
        x, P = kalman_xy(x, P, meas, R)
        result.append((x[:2]).tolist())
    kalman_x, kalman_y = zip(*result)

    if DEBUG_MODE:
        plt.plot(kalman_x, kalman_y, 'g-')
        plt.show()


if __name__ == "__main__":
    test_kalman_xy()
