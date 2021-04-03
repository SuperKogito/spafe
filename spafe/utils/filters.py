# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
from scipy.ndimage import sobel
from scipy.signal import gaussian


def gaussian_filter(M, std, sym=True):
    """
    Return a Gaussian window.
    """
    return gaussian(M, std, sym)


def sobel_filter(sig, axis=-1, mode="reflect", cval=0.0):
    """
    Calculate a Sobel filter.
    """
    return sobel(sig, axis, mode, cval)


def rasta_filter(x):
    """
    % y = rastafilt(x)
    %
    % rows of x = critical bands, cols of x = frame
    % same for y but after filtering
    %
    % default filter is single pole at 0.94
    """
    numer = np.arange(-2, 3)
    numer = (-1 * numer) / np.sum(numer * numer)
    denom = np.array([1, -0.94])

    zi = signal.lfilter_zi(numer, 1)
    y = np.zeros((x.shape))

    for i in range(x.shape[0]):
        y1, zi = signal.lfilter(numer, 1, x[i, 0:4], axis=0, zi=zi * x[i, 0])
        y1 = y1 * 0
        y2, _ = signal.lfilter(numer, denom, x[i, 4:x.shape[1]], axis=0, zi=zi)
        y[i, :] = np.append(y1, y2)
    return y


###################################################################################
# based on https://stackoverflow.com/questions/13901997/kalman-2d-filter-in-python
###################################################################################


def kalman_xy(x,
              P,
              measurement,
              R,
              motion=np.matrix('0. 0. 0. 0.').T,
              Q=np.matrix(np.eye(4))):
    """
    Args:
        x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
        P: initial uncertainty convariance matrix
        measurement: observed position
        R: measurement noise
        motion: external motion added to state vector x
        Q: motion noise (same shape as P)
    """
    return kalman(x,
                  P,
                  measurement,
                  R,
                  motion,
                  Q,
                  F=np.matrix('''
                      1. 0. 1. 0.;
                      0. 1. 0. 1.;
                      0. 0. 1. 0.;
                      0. 0. 0. 1.
                      '''),
                  H=np.matrix('''
                      1. 0. 0. 0.;
                      0. 1. 0. 0.'''))


def kalman(x, P, measurement, R, motion, Q, F, H):
    """
    Args:
        x: initial state
        P: initial uncertainty convariance matrix
        measurement: observed position (same shape as H*x)
        R: measurement noise (same shape as H)
        motion: external motion added to state vector x
        Q: motion noise (same shape as P)
        F: next state function: x_prime = F*x
        H: measurement function: position = H*x

    Returns:
        the updated and predicted new values for (x, P)

    See also http://en.wikipedia.org/wiki/Kalman_filter

    This version of kalman can be applied to many different situations by
    appropriately defining F and H.
    """
    # UPDATE x, P based on measurement m
    # distance between measured and current position-belief
    y = np.matrix(measurement).T - H * x
    S = H * P * H.T + R  # residual convariance
    K = P * H.T * S.I  # Kalman gain
    x = x + K * y
    I = np.matrix(np.eye(F.shape[0]))  # identity matrix
    P = (I - K * H) * P

    # PREDICT x, P based on motion
    x = F * x + motion
    P = F * P * F.T + Q

    return x, P


###################################################################################
