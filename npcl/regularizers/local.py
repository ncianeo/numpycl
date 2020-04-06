import pyopencl.array as cl_array
import numpy as np
from npcl.ops.local import grad2d, norm2d, divergence2d


def denoise_tv(image, weight=0.1, eps=2.e-4, n_iter_max=100):
    img_dev = image.copy()
    ndim = 2
    weight = np.float32(weight)
    eps = np.float32(eps)
    queue = img_dev.queue
    px = cl_array.zeros(queue, image.shape, dtype=np.float32)
    py = cl_array.zeros(queue, image.shape, dtype=np.float32)
    d = cl_array.zeros(queue, img_dev.shape, dtype=np.float32)
    tau = np.float32(1/(2.*ndim))
    N = np.float32(img_dev.shape[0]*img_dev.shape[1])
    i = 0
    while i < n_iter_max:
        if i > 0:
            # d will be the (negative) divergence of p
            d = divergence2d(px, py)
            d = -d
            out = img_dev + d
        else:
            out = img_dev
        E = cl_array.sum((d ** 2))
        # g stores the gradients of out along each axis
        # e.g. g[0] is the first order finite difference along axis 0
        gx, gy = grad2d(out)
        norm = norm2d(gx, gy)
        E += weight*cl_array.sum(norm)
        norm *= tau/weight
        norm += np.float32(1)
        px = px-tau*gx
        py = py-tau*gy
        px /= norm
        py /= norm
        E = E.get().item()
        E /= N
        if i == 0:
            E_init = E
            E_previous = E
        else:
            if np.abs(E_previous-E) < eps * E_init:
                break
            else:
                E_previous = E
        i += 1
    return out
