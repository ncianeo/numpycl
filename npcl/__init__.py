from .version import __version__
import pyopencl as cl
import pyopencl.clmath as cl_math
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


def zeros(shape, dtype):
    if ctx is None:
        create_ctx_queue()
    return cl_array.zeros(queue, shape, dtype)


def zeros_like(x):
    if ctx is None:
        create_ctx_queue()
    return cl_array.zeros_like(x)


def empty(shape, dtype):
    if ctx is None:
        create_ctx_queue()
    return cl_array.empty(queue, shape, dtype)


def empty_like(x):
    if ctx is None:
        create_ctx_queue()
    return cl_array.empty_like(x)


Array = cl_array.Array


# Conditionals
if_positive = cl_array.if_positive
maximum = cl_array.maximum
minimum = cl_array.minimum


# Reductions
sum = cl_array.sum
dot = cl_array.dot
vdot = cl_array.vdot
subset_dot = cl_array.subset_dot
max = cl_array.max
min = cl_array.min
subset_max = cl_array.subset_max
subset_min = cl_array.subset_min


# clmath functions
acos = cl_math.acos
acosh = cl_math.acosh
acospi = cl_math.acospi
asin = cl_math.asin
asinh = cl_math.asinh
asinpi = cl_math.asinpi
atan = cl_math.atan
atan2 = cl_math.atan2
atanh = cl_math.atanh
atanpi = cl_math.atanpi
atan2pi = cl_math.atan2pi
cbrt = cl_math.cbrt
ceil = cl_math.ceil
cos = cl_math.cos
cosh = cl_math.cosh
cospi = cl_math.cospi
erfc = cl_math.erfc
erf = cl_math.erf
exp = cl_math.exp
exp2 = cl_math.exp2
exp10 = cl_math.exp10
expm1 = cl_math.expm1
fabs = cl_math.fabs
floor = cl_math.floor
fmod = cl_math.fmod
frexp = cl_math.frexp
ilogb = cl_math.ilogb
ldexp = cl_math.ldexp
lgamma = cl_math.lgamma
log = cl_math.log
log2 = cl_math.log2
log10 = cl_math.log10
log1p = cl_math.log1p
logb = cl_math.logb
modf = cl_math.modf
nan = cl_math.nan
rint = cl_math.rint
round = cl_math.round
sin = cl_math.sin
sinh = cl_math.sinh
sinpi = cl_math.sinpi
sqrt = cl_math.sqrt
tan = cl_math.tan
tanh = cl_math.tanh
tanpi = cl_math.tanpi
tgamma = cl_math.tgamma
trunc = cl_math.trunc
