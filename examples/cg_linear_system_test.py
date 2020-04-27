import npcl
from npcl import to_device
import pyclblast
import numpy as np


matA = np.random.normal(size=(1600, 800)).astype(np.float32)
# make A symmetric
matA = matA.T@matA
vecx = np.random.normal(size=(800)).astype(np.float32)
vecb = np.dot(matA, vecx)

m, n = 800, 800

A_dev = to_device(matA)
b_dev = to_device(vecb)
x_init = to_device(np.zeros_like(vecx))

res = to_device(np.empty_like(vecb))


def A(x):
    buf = res.copy()
    pyclblast.gemv(
        npcl.queue, m, n, A_dev, x, buf, a_ld=n, alpha=1.0, beta=0.0,
        )
    return buf


x_cg, k = npcl.solvers.cg.solve_cg(A, b_dev, x_init, tol=1e-5, verbose=True)

print(
    'numerical error (root squared error / l2 norm of x): ',
    np.sqrt(((x_cg.get()-vecx)**2).sum())/np.sqrt((vecx**2).sum()),
)
print('true solution: ', vecx[:100])
print('numerical solution: ', x_cg.get()[:100])
