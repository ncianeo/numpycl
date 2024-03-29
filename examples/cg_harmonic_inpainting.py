from npcl import to_device
from npcl.solvers.inpaint import inpaint_h1
import numpy as np
import cv2
from time import time


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


mu = np.float32(1)
x_0 = to_device(np.zeros(img.shape))


start = time()
inpainted, iters = inpaint_h1(
    contaminated, mask, tol=1e-6,
    verbose=True,
    )

print('epalsed time:', time()-start)
cv2.imshow('inpainted', np.clip(255*inpainted.get(), 0, 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()