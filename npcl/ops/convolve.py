from os.path import abspath
import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np


prg = None


def build(parameter):
    if type(parameter) == cl.Context:
        ctx = parameter
    if type(parameter) == cl_array.Array:
        ctx = parameter.context
    global prg
    kernel_fp = abspath(__file__).replace('.py', '.cl')
    prg = cl.Program(ctx, open(kernel_fp, 'r').read())
    prg.build()


def convolve2d(x, k):
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
    queue = x.queue
    res = cl_array.empty(queue, x.shape, np.float32)
    prg.convolve2d(
        queue, x.shape, None,
        x.data, k.data, res.data,
        np.int32(k.shape[0]),
        np.int32(k.shape[1]),
        )
    return res


def convolve2d_sv(x, k):
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
    queue = x.queue
    res = cl_array.empty(queue, x.shape, np.float32)
    prg.convolve2d_sv(
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
    kernel = cl_array.empty_like(k)
    prg.transpose2d(
        queue, k.shape, None,
        k.data, kernel.data,
        )
    return kernel


def transpose2d_sv(k):
    if prg is None:
        build(k)
    queue = k.queue
    kernel = cl_array.empty_like(k)
    prg.transpose2d_sv(
        queue, k.shape[2:], None,
        k.data, kernel.data,
        np.int32(k.shape[0]), np.int32(k.shape[1]),
        )
    return kernel
