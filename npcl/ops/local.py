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


def grad2d(x):
    if prg is None:
        build(x)
    queue = x.queue
    gx = npcl.zeros_like(x)
    gy = npcl.zeros_like(x)
    prg.grad(queue, x.shape, None, x.data, gx.data, gy.data)
    return gx, gy


def norm2d(gx, gy):
    if prg is None:
        build(gx)
    queue = gx.queue
    norm = npcl.zeros_like(gx)
    prg.norm(queue, norm.shape, None, gx.data, gy.data, norm.data)
    return norm


def divergence2d(px, py):
    if prg is None:
        build(px)
    queue = px.queue
    d = npcl.zeros_like(px)
    prg.divergence2d(queue, d.shape, None, px.data, py.data, d.data)
    return d


def sign(x):
    return ((x >= 0).astype(np.float32)-0.5)*2


def soft_shrink(x, mu):
    shrinked = npcl.fabs(x)-mu
    return sign(x)*(shrinked > 0)*shrinked
