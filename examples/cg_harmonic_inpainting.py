import npcl
from npcl import to_device
import numpy as np
import cv2
from time import time

convolve = npcl.ops.convolve.convolve2d

img = cv2.imread('lake.tif', 0)/255
img = to_device(img)

mask = np.zeros(img.shape[:2], dtype=np.float32)
mask[256:257, :] = 1
mask[:, 256:257] = 1
mask = to_device(mask)

contaminated = img*(1-mask)
cv2.imshow('contaminated', np.clip(255*contaminated.get(), 0, 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

dx = np.array([[0, -1, 1]], dtype=np.float32)
dy = dx.T
dxT = dx[..., ::-1]
dyT = dxT.T
dx = to_device(dx)
dy = to_device(dy)
dxT = to_device(dxT)
dyT = to_device(dyT)

mu = np.float32(1)
x_0 = to_device(np.zeros(img.shape))


def masked_laplacian(x, mask):
    px = convolve(x, dx)
    py = convolve(x, dy)
    qx = convolve(mask*px, dxT)
    qy = convolve(mask*py, dyT)
    return qx+qy


def A(x):
    return (1-mask)*x+mu*masked_laplacian(x, mask)


start = time()
inpainted, iters = npcl.solvers.cg.solve_cg(
    A, contaminated, x_0, tol=1e-6,
    verbose=True,
    )

print('epalsed time:', time()-start)
cv2.imshow('inpainted', np.clip(255*inpainted.get(), 0, 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()