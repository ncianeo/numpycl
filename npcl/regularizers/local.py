import npcl
import numpy as np
from npcl.ops.local import grad2d, norm2d, divergence2d


def denoise_tv(image, weight=0.1, eps=2.e-4, n_iter_max=100):
    img_dev = image.copy()
    ndim = 2
    weight = np.float32(weight)
    eps = np.float32(eps)
    px = npcl.zeros_like(image)
    py = npcl.zeros_like(image)
    d = npcl.zeros_like(image)
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
        E = npcl.sum((d ** 2)).get()
        # (gx, gy) stores the gradients of out along each axis
        gx, gy = grad2d(out)
        norm = norm2d(gx, gy)
        E += weight*npcl.sum(norm).get()
        norm *= tau/weight
        norm += np.float32(1)
        px = px-tau*gx
        py = py-tau*gy
        px /= norm
        py /= norm
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
