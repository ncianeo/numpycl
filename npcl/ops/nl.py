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