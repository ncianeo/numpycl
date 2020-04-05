from os.path import abspath
import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np


prg = None


def build():
    from npcl import ctx, queue
    global ctx, queue
    if ctx is None:
        ctx = cl.create_some_context(interactive=False)
        queue = cl.CommandQueue(ctx)
    global prg
    kernel_fp = abspath(__file__).replace('.py', '.cl')
    prg = cl.Program(ctx, open(kernel_fp, 'r').read())
    prg.build()


def denoise_tv(image, weight=0.1, eps=2.e-4, n_iter_max=100):
    if prg is None:
        build()
    img_dev = image.copy()
    ndim = 2
    weight = np.float32(weight)
    eps = np.float32(eps)
    p = cl_array.zeros(queue, (ndim, ) + img_dev.shape, dtype=np.float32)
    g = cl_array.zeros_like(p)
    d = cl_array.zeros_like(img_dev)
    norm = cl_array.zeros_like(img_dev)
    tau = np.float32(1/(2.*ndim))
    N = np.float32(img_dev.shape[0]*img_dev.shape[1])
    i = 0
    while i < n_iter_max:
        if i > 0:
            # d will be the (negative) divergence of p
            prg.divergence(queue, img_dev.shape, None, p.data, d.data)
            d = -d
            out = img_dev + d
        else:
            out = img_dev
        E = cl_array.sum((d ** 2))
        # g stores the gradients of out along each axis
        # e.g. g[0] is the first order finite difference along axis 0
        prg.grad(queue, img_dev.shape, None, out.data, g.data)
        prg.norm(queue, img_dev.shape, None, g.data, norm.data)
        E += weight*cl_array.sum(norm)
        norm *= tau/weight
        norm += np.float32(1)
        p = p-tau*g
        prg.divide_3d_by_2d(queue, img_dev.shape, None, p.data, norm.data)
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
