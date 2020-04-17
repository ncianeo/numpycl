import pyopencl.array as cl_array
import numpy as np


def solve_fista(
        ATA, ATb, x_0, ProxR_solver,
        delta=np.float32(1.0), mu=np.float32(1e-3), tol=np.float32(1e-3),
        p=1, q=1, r=4, restarting=False,
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
        p, q, r : (np.float32) momentum parameter. 0<p<=1, 0<q<=1, 0<r<=4.
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
    bnorm = norms(ATb)

    def p_L(x):
        return ProxR_solver(x-delta*(ATA(x)-ATb), mu*delta)

    def norms(x):
        return cl_array.sum(x**2).get()

    if restarting:
        def restart(y, x, xold):
            return cl_array.sum((y-x)*(x-xold)).get() >= 0
        restarted = False

    while True:
        k += 1
        x = p_L(y)
        t = (p+np.sqrt(q+r*told**2))/2
        beta = np.float32(min((told-1)/t, 1))
        yold = y
        y = x + beta*(x-xold)
        seq_diff = norms(x-xold)
        if restarting:
            if restart(yold, x, xold):
                print(
                    'restarting occured at iteration ', k,
                )
                t = 1
                y = x
                if not restarted:
                    xi = ((4+beta)/5.)**(1/30.)
                    restarted = True
                r *= xi
        if np.isnan(seq_diff) or np.isinf(seq_diff):
            print('something wrong with the problem setting...')
            break
        if verbose is True:
            print(
                'iteration number: ', k, ', sequential difference: ',
                np.sqrt(seq_diff/bnorm),
                )
        if seq_diff < bnorm*tol**2:
            break
        if k == max_iter:
            break
        xold = x.copy()
        told = t
    return x, k


def solve_gfista(
        ATA, ATb, x_0, ProxR_solver,
        L_inv=np.float32(1.0), delta=np.float32(1.),
        eta=np.float32(0.96), S=np.float32(1.),
        mu=np.float32(1e-3), tol=np.float32(1e-3),
        verbose=False, max_iter=50,
        ):
    """
    Greedy FISTA Method.

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
        L_inv : (np.float32) inverse of Lipschitz constant of A^TA.
        delta : (np.float32) starting step size.
        mu : (np.float32) regularization parameter.
        tol : (np.float32) represents tolerence value.
        eta : (np.float32) shrink of step size.
        S : (np.float32) safeguard.
        max_iter : maximum number of iterations.

    Outputs:
        y : (pyopencl.array.Array) the solution x.
        k : (int) the total iteration number.

    Reference:
        - A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems
        Read More: https://epubs.siam.org/doi/abs/10.1137/080716542
        - Improving "Fast Iterative Shrinkage-Thresholding Algorithm": Faster, Smarter and Greedier
        Read More: https://arxiv.org/abs/1811.01430
    """

    x = x_0.copy()
    y = x.copy()
    delta0 = np.float32(delta*L_inv)
    k = 0
    bnorm = norms(ATb)

    def p_L(x, delta):
        return ProxR_solver(x-delta*(ATA(x)-ATb), mu*delta)

    def norms(x):
        return cl_array.sum(x**2).get()

    def restart(y, x, xold):
        return cl_array.sum((y-x)*(x-xold)).get() >= 0

    while True:
        k += 1
        xold = x
        yold = y

        x = p_L(y, delta)
        y = x + x-xold
        if restart(yold, x, xold):
            y = xold
            if verbose:
                print('restarting occured at iteration ', k)
        seq_diff = norms(x-xold)
        if k == 1:
            safeguard = S*norms(x-xold)
        elif seq_diff > safeguard:
            delta = np.float32(max(eta*delta, delta0))
            if verbose:
                print(
                    'safeguard activated at iteration', k,
                    'delta now: ', delta,
                    )
        if np.isnan(seq_diff) or np.isinf(seq_diff):
            print('something wrong with the problem setting...')
            break
        if verbose:
            print(
                'iteration number: ', k, ', sequential difference: ',
                np.sqrt(seq_diff/bnorm),
                )
        if seq_diff < bnorm*tol**2:
            break
        if k == max_iter:
            break
    return x, k
