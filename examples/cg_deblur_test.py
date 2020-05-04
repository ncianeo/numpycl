import npcl
from npcl import to_device
import numpy as np
import cv2
from time import time

convolve = npcl.ops.convolve.convolve2d

img = cv2.imread('lake.tif', 0)/255
img = to_device(img)

# 5 x 5 box kernel
kernel = np.ones((5, 5))/25.
kernel = to_device(kernel)

blurry = convolve(img, kernel)
x_0 = to_device(np.zeros(img.shape))

cv2.imshow('blurry', np.clip(255*blurry.get(), 0, 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()


def A(x):
    return convolve(x, kernel)


start_time = time()
deblurred, iters = npcl.solvers.cg.solve_cg(
    A, blurry, x_0, tol=5e-4,
    verbose=True,
    )

print('time elapsed:', time()-start_time)
cv2.imshow('deblurred', np.clip(255*deblurred.get(), 0, 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
