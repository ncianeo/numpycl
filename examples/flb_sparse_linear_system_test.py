import npcl
from npcl import to_device
import pyclblast
import numpy as np
from time import time


m, n = 1200, 4000

# Prepare random matrix A
matA = np.random.normal(size=(m, n)).astype(np.float32)

# Prepare random sparse vector x
mask = np.random.uniform(size=(n)).astype(np.float32)
mask = mask > 0.05
vecx = np.random.uniform(-1, 1, size=(n)).astype(np.float32)
vecx[mask] *= 0

vecb = np.dot(matA, vecx)

A_dev = to_device(matA)
AT_dev = to_device(matA.T)
b_dev = to_device(vecb)


def A(x):
    res = to_device(np.zeros_like(vecb))
    pyclblast.gemv(
        npcl.queue, m, n, A_dev, x, res, a_ld=n, alpha=1.0, beta=0.0,
        )
    return res


def AT(x):
    res = to_device(np.zeros_like(vecx))
    pyclblast.gemv(
        npcl.queue, n, m, AT_dev, x, res, a_ld=m, alpha=1.0, beta=0.0,
        )
    return res


start = time()

x_flb, k = npcl.solvers.flb.solve_flb(
    A, AT, b_dev, stuck=1e-1, tol=1e-6,
    verbose=True, max_iter=50000,
    )

print('elapsed time: ', time()-start)
print(
    'numerical error (absolute error / l1 norm of x): ',
    np.abs(x_flb.get()-vecx).sum()/np.abs(vecx).sum(),
)
print('true solution: ', vecx[:100])
print('numerical solution: ', x_flb.get()[:100])
