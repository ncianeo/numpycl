import pyopencl.array as cl_array
import numpy as np


def solve_fbs(
        ATA, ATb, x_0, ProxR_solver,
        delta=np.float32(0.9), mu=np.float32(1e-3), tol=np.float32(1e-3),
        verbose=False, max_iter=50,
        ):
    """
    Forward-Backward Splitting (FBS) Method.

    This function solves the following problem:
        f(x) = 1/2*|Ax-b|^2 + mu*R(x).

    Solving FBS needs computation of A^TA, A^Tb.
    Therefore we set inputs of the function as follows.

    Inputs:
        ATA : a python function that computes A^TAx, i.e.,
            ATA(x) = A^TAx
            for a vector (pyopencl.array.Array) x.
        ATb : (pyopencl.array.Array) represents the vector A^tb.
        x_0 : (pyopencl.array.Array) represents the initial point x_0.
        ProxR_solver : a python function that computes Prox_{mu R}(x), i.e.,
            ProxR_solver(x, mu) = Prox_{mu R}(x)
            for a vector (pyopencl.array.Array) x and a scalar mu > 0.
        delta : (np.float32) parameter for gradient update step.
        mu : (np.float32) regularization parameter.
        tol : (np.float32) represents tolerence value.
        max_iter : maximum number of iterations.

    Outputs:
        x : (pyopencl.array.Array) the solution x.
        k : (int) the total iteration number.
    """
    x = x_0.copy()
    k = 0
    while True:
        k += 1
        v = x - delta*(ATA(x)-ATb)
        x_new = ProxR_solver(v, delta*mu)
        if cl_array.sum((x_new-x)**2).get() < cl_array.sum(x**2).get()*tol**2:
            break
        if k == max_iter:
            break
        if verbose is True:
            print('iteration number: ', k)
        x = x_new.copy()
    return x_new, k
