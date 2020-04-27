import npcl
from npcl.ops.local import sign, soft_shrink
import numpy as np


def solve_flb(
        A, AT, b,
        delta=np.float32(5e-4), mu=np.float32(2e4), tol=np.float32(1e-5),
        stuck=np.float32(1e-2), verbose=False, max_iter=5000,
        ):
    """
    Fast Linearized Bregman (FLB) Method.

    This function solves the following problem:
        minimize |x|_1 subject to Ax=b

    Inputs:
        ATA : a python function that computes A^TAx, i.e.,
            ATA(x) = A^TAx
            for a vector (pyopencl.array.Array) x.
        AT : a python function that computes ATx, i.e.,
            AT(x) = ATx
            for a vector (pyopencl.array.Array) x.
        b : (pyopencl.array.Array) represents the vector b.
        delta : (np.float32) parameter for gradient update step.
        mu : (np.float32) l1 regularization parameter.
        tol : (np.float32) represents tolerence value.
        stuck: (np.float32) represents stuck constant
        max_iter : maximum number of iterations.

    Outputs:
        x : (pyopencl.array.Array) the solution x.
        k : (int) the total iteration number.
    """
    ATb = AT(b)

    def ATA(x):
        return AT(A(x))

    def norm(x):
        return npcl.sum(npcl.fabs(x)).get()

    x = npcl.zeros_like(ATb)
    v = x.copy()
    normb = norm(b)
    kick_test = [0 for i in range(5)]
    k = 0
    while True:
        k += 1
        v += (ATb-ATA(x))
        x_new = delta*soft_shrink(v, mu)
        residual = np.log(norm(A(x_new)-b))
        if verbose is True:
            print('iteration number: ', k, 'log residual: ', residual)
        kick_test = kick_test[-4:] + [residual]
        if min(kick_test)+stuck > max(kick_test):
            k += 1
            I_0 = (x_new == 0.).astype(np.float32)
            r = ATb-ATA(x_new)
            s = ((mu*sign(r)-v)/r).astype(np.int32)*I_0
            smin = np.float32(npcl.min(s + 1e7*(1-I_0)).get())
            if smin <= 2:
                v += r
            else:
                v += smin*I_0*r
            x_new = delta*soft_shrink(v, mu)
            if verbose is True:
                print(
                    'kicking occured at iteration number: ',
                    k, 'log residual: ', residual,
                    )
        r_new = A(x_new) - b
        if norm(r_new) < normb*tol:
            break
        if k == max_iter:
            break
        x = x_new.copy()
    return x_new, k
