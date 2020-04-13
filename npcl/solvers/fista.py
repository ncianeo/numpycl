import pyopencl.array as cl_array
import numpy as np


def solve_fista(
        ATA, ATb, x_0, ProxR_solver,
        delta=np.float32(1.0), mu=np.float32(1e-3), tol=np.float32(1e-3),
        verbose=False, max_iter=50,
        ):
    """
    Fast Iterative Shrinkage Thresholding Algorithm (FISTA) Method.

    This function solves the following convex problem with a linear operator A:
        f(x) = 1/2*|Ax-b|^2 + mu*R(x).

    Solving FISTA needs computation of A^TA, A^Tb.
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
        delta : (np.float32) inverse of Lipschitz constant of A^TA.
        mu : (np.float32) regularization parameter.
        tol : (np.float32) represents tolerence value.
        max_iter : maximum number of iterations.

    Outputs:
        y : (pyopencl.array.Array) the solution x.
        k : (int) the total iteration number.

    Reference:
        A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems
        Read More: https://epubs.siam.org/doi/abs/10.1137/080716542
    """

    xold = x_0.copy()
    y = xold.copy()
    told = np.float32(1.)
    k = 0

    # def p(x):
    #     print(x)
    #     v = x-delta*(ATA(x)-ATb)
    #     print(v)
    #     return ProxR_solver(v, mu*delta)

    def norms(x):
        return cl_array.sum(x**2).get()

    while True:
        k += 1
        v = y-delta*(ATA(y)-ATb)
        print(v)
        x = ProxR_solver(v, mu*delta)
        # x = p(y)
        print(x.get())
        t = (1+np.sqrt(1+4*told**2))/2
        y = x + (told-1)/t*(x-xold)
        print(y.get())
        # print(norms(y))
        if norms((x-xold)**2) < norms(xold**2)*tol**2:
            break
        if verbose is True:
            print('iteration number: ', k)
        if k == max_iter:
            break
        xold = x.copy()
        told = t
    return y, k
