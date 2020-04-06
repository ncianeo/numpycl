from .version import __version__
import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np

ctx = None
queue = None

from . import ops, regularizers, solvers


def create_ctx_queue():
    global ctx, queue
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)


def to_device(x, dtype=np.float32):
    if ctx is None:
        create_ctx_queue()
    return cl_array.to_device(queue, np.require(x, dtype, 'C'))
