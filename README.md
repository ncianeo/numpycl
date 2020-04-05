# numpycl
=====
Numerical Library Written in Python with PyOpenCL

## Requirements
=====
1. Python >= 3.6
2. PyOpenCL
3. Numpy

## Installation
======
```bash
pip install -r requirements.txt
```

## Usage Examples
======
* Conjugate Gradient Solver (cg_deblur_test.py)
```python
import npcl
from npcl import to_device
import numpy as np
import cv2

convolve = npcl.ops.convolve.convolve2d

img = cv2.imread('lake.tif', 0)
img = to_device(img)

# 5 x 5 box kernel
kernel = np.ones((5, 5))/25.
kernel = to_device(kernel)

blurry = convolve(img, kernel)
x_0 = to_device(np.zeros(img.shape))

cv2.imshow('blurry', np.clip(blurry.get(), 0, 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()


def A(x):
    return convolve(x, kernel)


deblurred, iters = npcl.solvers.cg.solve_cg(
    A, blurry, x_0, verbose=True,
    )

cv2.imshow('deblurred', np.clip(deblurred.get(), 0, 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
```

* Forward-Backward Splitting Method with Total Variation Regularizer (fbs_deblur_test.py)
```python
import npcl
from npcl import to_device
import numpy as np
import cv2

convolve = npcl.ops.convolve.convolve2d

img = cv2.imread('lake.tif', 0)
img = to_device(img)

# noise will be added
noise = 5*np.random.normal(size=img.shape)
noise = to_device(noise)

# 5 x 5 box kernel
kernel = np.ones((5, 5))/25.
kernel = to_device(kernel)

blurry = convolve(img, kernel)+noise

cv2.imshow('blurry', np.clip(blurry.get(), 0, 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

deblurred, iter = npcl.solvers.deconv.deconv_fbstv(
    blurry, kernel, mu=1e-1, verbose=True,
    )

cv2.imshow('deblurred', np.clip(deblurred.get(), 0, 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
```