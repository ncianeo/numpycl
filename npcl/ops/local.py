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


def grad2d(x):
    if prg is None:
        build(x)
    queue = x.queue
    gx = cl_array.zeros(queue, x.shape, dtype=np.float32)
    gy = cl_array.zeros(queue, x.shape, dtype=np.float32)
    prg.grad(queue, x.shape, None, x.data, gx.data, gy.data)
    return gx, gy


def norm2d(gx, gy):
    if prg is None:
        build(gx)
    queue = gx.queue
    norm = cl_array.zeros(queue, gx.shape, dtype=np.float32)
    prg.norm(queue, norm.shape, None, gx.data, gy.data, norm.data)
    return norm


def divergence2d(px, py):
    if prg is None:
        build(px)
    queue = px.queue
    d = cl_array.zeros(queue, px.shape, dtype=np.float32)
    prg.divergence2d(queue, d.shape, None, px.data, py.data, d.data)
    return d
