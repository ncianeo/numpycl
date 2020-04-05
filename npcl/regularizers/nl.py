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