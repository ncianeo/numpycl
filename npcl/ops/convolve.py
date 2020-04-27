from os.path import abspath
import pyopencl as cl
import npcl
import numpy as np


prg = None


def build(parameter):
    if type(parameter) == cl.Context:
        ctx = parameter
    if type(parameter) == npcl.Array:
        ctx = parameter.context
    global prg
    kernel_fp = abspath(__file__).replace('.py', '.cl')
    prg = cl.Program(ctx, open(kernel_fp, 'r').read())
    prg.build()


def convolve2d(x, k, padding='zero'):
    """
    Compute 2D convolution.

    This function computes
        y = (k*x),
    where k : convolutional kernel.

    Inputs:
        x : input array.
        k : convolutional kernel array.

    Outputs:
        y : output array.
    """
    if prg is None:
        build(x)
    if padding == 'zero':
        run_kernel = prg.convolve2d_z
    elif padding == 'same':
        run_kernel = prg.convolve2d_s
    elif padding == 'wrap':
        run_kernel = prg.convolve2d_w
    queue = x.queue
    res = npcl.zeros_like(x)
    run_kernel(
        queue, x.shape, None,
        x.data, k.data, res.data,
        np.int32(k.shape[0]),
        np.int32(k.shape[1]),
        )
    return res


def convolve2d_sv(x, k, padding='zero'):
    r"""
    Compute 2D spatially-variant convolution.

    This function computes
        y_{i, j} = \sum_{k, l} k_{k, l, i, j} x_{i + k, j + l},
    where k : convolutional kernel.

    Inputs:
        x : input array (2D).
        k : convolutional kernel array (4D).
            dimensions : kernel window (2D) x image size (2D)

    Outputs:
        y : output array.
    """
    if prg is None:
        build(x)
    if padding == 'zero':
        run_kernel = prg.convolve2d_sv_z
    elif padding == 'same':
        run_kernel = prg.convolve2d_sv_s
    elif padding == 'wrap':
        run_kernel = prg.convolve2d_sv_w
    queue = x.queue
    res = npcl.zeros_like(x)
    run_kernel(
        queue, x.shape, None,
        x.data, k.data, res.data,
        np.int32(k.shape[0]),
        np.int32(k.shape[1]),
        )
    return res


def transpose2d(k):
    if prg is None:
        build(k)
    queue = k.queue
    kernel = npcl.empty_like(k)
    prg.transpose2d(
        queue, k.shape, None,
        k.data, kernel.data,
        )
    return kernel


def transpose2d_sv(k):
    if prg is None:
        build(k)
    queue = k.queue
    kernel = npcl.empty_like(k)
    prg.transpose2d_sv(
        queue, k.shape[2:], None,
        k.data, kernel.data,
        np.int32(k.shape[0]), np.int32(k.shape[1]),
        )
    return kernel
