import numpy as np
from npcl.ops.convolve import convolve2d, transpose2d
from npcl.ops.convolve import convolve2d_sv, transpose2d_sv
from npcl.regularizers.local import denoise_tv
from .fbs import solve_fbs
from .fista import solve_fista


def deconv_fbs(
        img_blurry, psf, kernel=None, denoiser=denoise_tv,
        mu=np.float32(1e-3), tol=np.float32(1e-3), delta=np.float32(0.9),
        max_iter=50, verbose=False,
        ):
    """
    Deconvolution with Forward-Backward Splitting Method.
    """
    x_0 = img_blurry.copy()
    if kernel is None:
        kernel = transpose2d(psf)
    mu = np.float32(mu)
    delta = np.float32(delta)
    tol = np.float32(tol)
    ATb = convolve2d(x_0, psf)

    def ATA(x):
        return convolve2d(convolve2d(x, kernel), psf)
    ProxR = denoiser

    x, k = solve_fbs(
        ATA, ATb, x_0, ProxR,
        delta=delta, mu=mu, tol=tol, verbose=verbose, max_iter=max_iter,
        )
    return x, k


def deconv_fista(
        img_blurry, psf, kernel=None, denoiser=denoise_tv,
        mu=np.float32(1e-3), tol=np.float32(1e-3), delta=np.float32(0.9),
        max_iter=50, verbose=False,
        ):
    """
    Deconvolution with FISTA method.
    """
    x_0 = img_blurry.copy()
    if kernel is None:
        kernel = transpose2d(psf)
    mu = np.float32(mu)
    delta = np.float32(delta)
    tol = np.float32(tol)
    ATb = convolve2d(x_0, psf)

    def ATA(x):
        return convolve2d(convolve2d(x, kernel), psf)
    ProxR = denoiser

    x, k = solve_fista(
        ATA, ATb, x_0, ProxR,
        delta=delta, mu=mu, tol=tol, verbose=verbose, max_iter=max_iter,
        )
    return x, k


def deconv_sv_fbstv(
        img_blurry, psf, kernel=None,
        mu=np.float32(1e-3), tol=np.float32(1e-3), delta=np.float32(0.9),
        max_iter=50, verbose=False,
        ):
    x_0 = img_blurry.copy()
    if kernel is None:
        kernel = transpose2d_sv(psf)
    mu = np.float32(mu)
    delta = np.float32(delta)
    tol = np.float32(tol)
    ATb = convolve2d_sv(x_0, psf)

    def ATA(x):
        return convolve2d_sv(convolve2d_sv(x, kernel), psf)
    ProxR = denoise_tv

    x, k = solve_fbs(
        ATA, ATb, x_0, ProxR,
        delta=delta, mu=mu, tol=tol, verbose=verbose, max_iter=max_iter,
        )
    return x, k
