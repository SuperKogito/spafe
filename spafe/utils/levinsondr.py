"""
from:
    https://github.com/RJTK/Levinson-Durbin-Recursion/blob/master/levinson/levinson.py

Implementation of Levinson Recursion and other associated routines,
particularly the Block Toeplitz versions of Whittle and Akaike.
"""
import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def lev_durb(r):
    """
    Comsumes a length p + 1 vector r = [r(0), ..., r(p)] and returns
    (a, G, eps) as follows:

    Args:
        r (numpy array) : input vector.

    Returns:
        - a (np.array): Length p + 1 array (with a[0] = 1.0) consisting
        of the filter coefficients for an all-pole model of a signal
        having autocovariance r.
        - G (np.array): Length p array of reflection coefficients.
        It is guaranteed that `\|G[tau]\| <= 1`.
        - eps (np.array): The sequence of errors achieved by all-pole
        models of progressively larger order.  eps is guaranteed to
        satisfy eps >= 0.

    NOTE:
        We don't handle complex data
        We get a solution to the system R @ a = eps * e1 where R = toep(r)
        and e1 is the first canonical basis vector.  The variables are a[1:]
        and eps.  NOTE: For epsilon we are returning a sequence of errors for
        progressively larger systems.
        One of the key advantages of this algorithm is that the resulting
        filter is guaranteed to be stable, and the prediction error is directly
        available as a byproduct.
        Moreover, the sequence G has the property forall tau: `\|G(tau)\| < 1.0`
        if and only if r is a positive definite covariance sequence.

    Reference:
        @book{hayes2009statistical,
        title={Statistical digital signal processing and modeling},
        author={Hayes, Monson H},
        year={2009},
        publisher={John Wiley \& Sons}}
    """
    # Initialization
    p      = len(r) - 1
    a      = np.zeros(p + 1)
    a[0]   = 1.0
    G      = np.zeros(p)
    eps    = np.zeros(p + 1)
    eps[0] = r[0]

    for tau in range(p):
        # Compute reflection coefficient
        conv = r[tau + 1]
        for s in range(1, tau + 1):
            conv = conv + a[s] * r[tau - s + 1]
        G[tau] = -conv / eps[tau]

        # Update 'a' vector
        a_cpy = np.copy(a)
        for s in range(1, tau + 1):
            a_cpy[s] = a[s] + G[tau] * np.conj(a[tau - s + 1])
        a = a_cpy
        a[tau + 1]   = G[tau]
        eps[tau + 1] = eps[tau] * (1 - np.abs(G[tau])**2)
    return a, G, eps


@numba.jit(nopython=True, cache=True)
def _whittle_lev_durb(R):
    p = len(R) - 1
    n = R[0].shape[0]

    A = np.zeros((p + 1, n, n))
    A_bar = np.copy(A)  # Backward coeffs

    A[0] = np.eye(n)
    A_bar[0] = np.eye(n)

    Sigma = np.zeros((p + 1, n, n))  # Forward error variance
    Sigma_bar = np.zeros((p + 1, n, n))  # Backward error variance

    Delta = np.zeros((p + 1, n, n))  # (Partial) Reflection coefficients
    Delta_bar = np.zeros((p + 1, n, n))
    Delta[0] = A[0]
    Delta_bar[0] = A_bar[0]

    Sigma[0] = R[0]
    Sigma_bar[0] = R[0]

    for k in range(p):
        Delta[k + 1] = np.zeros((n, n))
        Delta_bar[k + 1] = np.zeros((n, n))

        for tau in range(k + 1):
            Delta[k + 1] = Delta[k + 1] + A[tau] @ R[k - tau + 1]
            Delta_bar[k + 1] = Delta_bar[k + 1] + A_bar[tau] @ R[k - tau + 1].T

        A_cpy = np.copy(A)
        A_bar_cpy = np.copy(A_bar)

        # These are the real reflection coefficients
        A_cpy[k + 1] = -np.linalg.solve(
            Sigma_bar[k], Delta[k + 1].T).T
        A_bar_cpy[k + 1] = -np.linalg.solve(
            Sigma[k], Delta_bar[k + 1].T).T

        for tau in range(1, k + 1):
            A_cpy[tau] = A[tau] + A_cpy[k + 1] @ A_bar[k - tau + 1]
            A_bar_cpy[tau] = A_bar[tau] + A_bar_cpy[k + 1] @ A[k - tau + 1]

        A = np.copy(A_cpy)
        A_bar = np.copy(A_bar_cpy)

        Sigma[k + 1] = Sigma[k] + A[k + 1] @ Delta_bar[k + 1]
        Sigma_bar[k + 1] = Sigma_bar[k] + A_bar[k + 1] @ Delta[k + 1]

    return A, A_bar, Delta, Delta_bar, Sigma, Sigma_bar


@numba.jit(nopython=True, cache=True)
def whittle_lev_durb(R):
    """
    Comsumes a length p + 1 vector R = [R(0), ..., R(p)] of n x n
    block matrices which must be a valid (vector-)autocovariance sequence
    (i.e. the block-toeplitz matrix formed from R must be positive
    semi-definite) and returns (A, G, S) as follows:

    Args:
        R (numpy array) : input vector

    Returns:
        - A (List[np.array]): Length p + 1 array (with a[0] = np.eye(n))
        consisting of the filter coefficients for an all-pole model of a
        signal having autocovariance R(tau).
        - G (List[np.array]): Length p list of reflection coefficient matrices.
        - S (np.array): The variance matrix achieved by the all-pole
        model.  S is guaranteed to be positive semi-definite

    Note:

    We are returning a solution to: block-toep(R) @ A = e1 (x) S where (x)
    denote kronecker product and e1 is the first canonical basis vector.
    The (matrix-)variables are A[1:] and S.
    Fortunately, the block version of this algorithm also enjoys the
    stability property of the scalar version, i.e. det `\|A(z)\|` has it's
    zeros within the unit circle.
    """
    A, _, Delta, _, V, _ = _whittle_lev_durb(R)
    return A, Delta, V


@numba.jit(nopython=True, cache=True)
def reflection_coefs(Delta, Delta_bar, Sigma, Sigma_bar):
    """
    Calculates the reflection coefficients
    G[tau] = -Delta[tau] @ Sigma_bar[tau]^-1
    G_bar[tau] = -Delta_bar[tau] @ Sigma[tau]^-1
    """
    p, n, _ = Sigma.shape
    G = np.empty((p, n, n))
    G_bar = np.empty((p, n, n))

    G[0] = Sigma[0]
    G_bar[0] = Sigma_bar[0]
    for k in range(p - 1):
        G[k + 1] = -np.linalg.solve(
            Sigma_bar[k], Delta[k + 1].T).T
        G_bar[k + 1] = -np.linalg.solve(
            Sigma[k], Delta_bar[k + 1].T).T
    return G, G_bar


@numba.jit(nopython=True, cache=True)
def partial_autocovariance(R):
    """
    Obtains the partial autocovariance sequence from the WLD recursion
    """
    _, _, _, Delta_bar, _, _ = _whittle_lev_durb(R)
    return Delta_bar


@numba.jit(nopython=True, cache=True)
def fit_model_ret_plac(R):
    """
    A function which returns the coefficients B for a VAR(p) model
    as well as the sequence of partialautocorrelation matrices.
    Essentially this was crafted entirely to construct some particular
    LASSO weights.
    """
    A, _, _, Delta_bar, V, V_bar = _whittle_lev_durb(R)
    p = len(R) - 1

    Wplac = np.empty_like(Delta_bar)
    for k in range(p + 1):
        S_bar = 1. / np.sqrt(np.diag(V_bar[k]))
        S = 1. / np.sqrt(np.diag(V[k]))
        # Wplac[k] = S_bar[:, None] * Delta_bar[k] * S[None, :]  # no numba
        Wplac[k] = S_bar.reshape((-1, 1)) * Delta_bar[k]
        Wplac[k] = Wplac[k] * S.reshape((1, -1))
    B = A_to_B(A)
    return B, Wplac


@numba.jit(nopython=True, cache=True)
def step_up(G, G_bar):
    """
    The coefficients A_p(p) are particularly important for the
    levinson durbin recursion and are often referred to as
    reflection coefficients.  They are enough to characterize
    the whole of the sequence of coefficients.
    """
    p = len(G) - 1
    n = G.shape[1]
    A = np.empty((p + 1, n, n))
    A_bar = np.copy(A)  # Backward coeffs

    A[0] = np.eye(n)
    A_bar[0] = np.eye(n)

    A_cpy = np.copy(A)
    A_bar_cpy = np.copy(A_bar)

    for k in range(p):
        A_cpy = np.copy(A)
        A_bar_cpy = np.copy(A_bar)
        A_cpy[k + 1] = G[k + 1]
        A_bar_cpy[k + 1] = G_bar[k + 1]

        for tau in range(1, k + 1):
            A_cpy[tau] = A[tau] + A_cpy[k + 1] @ A_bar[k - tau + 1]
            A_bar_cpy[tau] = A_bar[tau] + A_bar_cpy[k + 1] @ A[k - tau + 1]
        A = np.copy(A_cpy)
        A_bar = np.copy(A_bar_cpy)
    return A, A_bar


@numba.jit(nopython=True, cache=True)
def compute_covariance(X, p_max):
    """
    Estimates covariances of X and returns an n x n x p_max array.
    The covariance sequence is guaranteed to be positive semidefinite.
    """
    T, n = X.shape
    R = np.empty((p_max + 1, n, n))
    R[0] = X.T @ X / T
    for tau in range(1, p_max + 1):
        R[tau] = X[tau:, :].T @ X[: -tau, :] / T
    return R


@numba.jit(nopython=True, cache=True)
def yule_walker(A, R):
    """
    Computes YW(A, R)(s) = sum_{tau = 0}^p A(tau) R(s - tau) for s = 0, ..., p
    We should have YW(A, R)(0) = V and YW(A, R)(s) = 0 for s != 0.
    p = len(A) - 1
    """
    p = len(A) - 1
    if len(A.shape) == 1:
        n = 1
    else:
        n = A.shape[1]

    YW = np.zeros((p + 1, n, n))

    for k in range(p + 1):
        for tau in range(p + 1):
            if k - tau >= 0:
                YW[k] += A[tau] @ R[k - tau]
            else:
                YW[k] += A[tau] @ R[tau - k].T
    return YW


@numba.jit(nopython=True, cache=True)
def A_to_B(A):
    p = len(A) - 1
    n = A.shape[1]
    B = np.empty((p, n, n))
    for tau in range(p):
        B[tau] = -A[tau + 1]
    return B


@numba.jit(nopython=True, cache=True)
def B_to_A(B):
    p = len(B)
    n = B.shape[1]
    A = np.empty((p + 1, n, n))
    A[0] = np.eye(n)
    for tau in range(1, p + 1):
        A[tau] = -B[tau - 1]
    return A


def block_companion(B):
    """
    Produces a block companion from the matrices B[0], B[1], ... , B[p - 1]
    [B0, B1, B2, ... Bp-1]
    [ I,  0,  0, ... 0   ]
    [ 0,  I,  0, ... 0   ]
    [ 0,  0,  I, ... 0   ]
    [ 0,  0, ..., I, 0   ]
    """
    p = len(B)
    B = np.hstack([B[k] for k in range(p)])  # The top row
    n = B.shape[0]

    I = np.eye(n * (p - 1))
    Z = np.zeros((n * (p - 1), n))
    R = np.hstack((I, Z))
    B = np.vstack((B, R))
    return B


def system_rho(B):
    """
    Computes the syste stability coefficient for an all-pole
    system with coefficient matrices B[0], B[1], ...
    """
    C = block_companion(B)
    ev = np.linalg.eigvals(C)
    return max(abs(ev))


def is_stable(B):
    rho = system_rho(B)
    return rho < 1
