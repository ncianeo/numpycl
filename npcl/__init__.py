import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np

ctx = None
queue = None

from . import ops, regularizers, solvers


def to_device(x, dtype=np.float32):
    global ctx, queue
    if ctx is None:
        ctx = cl.create_some_context(interactive=False)
        queue = cl.CommandQueue(ctx)
    return cl_array.to_device(queue, np.require(x, dtype, 'C'))
