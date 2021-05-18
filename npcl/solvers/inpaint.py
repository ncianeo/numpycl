import numpy as np
import npcl
from npcl.ops.convolve import convolve2d as convolve
from npcl import to_device
from .cg import solve_cg


dx = np.array([[0, -1, 1]], dtype=np.float32)
dy = dx.T
dxT = dx[..., ::-1]
dyT = dxT.T
dx = to_device(dx)
dy = to_device(dy)
dxT = to_device(dxT)
dyT = to_device(dyT)


def inpaint_h1(
        img, mask,
        mu=np.float32(1), tol=np.float32(1e-4),
        max_iter=1000, verbose=False,
        ):
    """
    Harmonic Inpainting with conjugate gradient method.
    """
    mu = np.float32(mu)

    def masked_laplacian(x, mask):
        px = convolve(x, dx)
        py = convolve(x, dy)
        qx = convolve(mask*px, dxT)
        qy = convolve(mask*py, dyT)
        return qx+qy

    def A(x):
        return (1-mask)*x+mu*masked_laplacian(x, mask)

    x_0 = img.copy()
    b = img * (1-mask)
    
    tol = np.float32(tol)

    x, k = solve_cg(
        A, b, x_0, tol=tol, verbose=verbose, max_iter=max_iter,
        )
    return x, k


def inpaint_tv(img, mask,
        mu=np.float32(1e-2), gamma=np.float32(1e-1), tol=np.float32(1e-4),
        max_iter=1000, verbose=False,
        ):
    """
    Total Variation Inpainting with Split Bregman method.
    """
    
    def masked_laplacian(x):
        px = convolve(x, dx)
        py = convolve(x, dy)
        qx = convolve(mask*px, dxT)
        qy = convolve(mask*py, dyT)
        return qx+qy

    def masked_divergence(px, py):
        qx = convolve(mask*px, dxT)
        qy = convolve(mask*py, dyT)
        return qx+qy

    def A(x):
        return (1-mask)*x + gamma*masked_laplacian(x)


    d1, d2, b1, b2, uf, ul = [npcl.zeros_like(img) for _ in range(6)]
    img_norm = npcl.sum(img**2)

    for k in range(max_iter):
        # u-subproblem
        b = (1-mask) * img + gamma*masked_divergence(d1-b1, d2-b2)
        ul, k_sub = solve_cg(A, b, uf, max_iter=10)
        # d-subproblem
        u1 = convolve(ul, dx)
        u2 = convolve(ul, dy)
        d1, d2 = shrink(u1+b1, u2+b2, mu*mask/gamma)
        b1 = b1 + u1 - d1
        b2 = b2 + u2 - d2
        gap = npcl.sum((ul-uf)**2) / img_norm
        if verbose is True:
            print(
                'iteration number: ', k+1,
                ', gap: ', gap.get(),
                )
        if gap < tol**2:
            break
        uf = ul.copy()
    return uf, k


def shrink(ax, ay, mu):
    eps = np.finfo(np.float32).eps
    norm = npcl.sqrt(ax**2 + ay**2)
    sx = ax / (norm + eps)
    sy = ay / (norm + eps)
    modular = (norm - mu)
    mask = (modular > 0).astype('float32')
    return modular*mask*sx, modular*mask*sy
