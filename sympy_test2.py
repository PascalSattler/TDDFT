# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 11:59:24 2022

@author: Pascal Sattler
"""

import numpy as np
from sympy import exp, log
from sympy.vector import CoordSys3D
from sympy.vector import divergence as div

S = CoordSys3D('S')

A = (S.x**2 * S.z) * S.i + (exp(2*S.y)) * S.j + (log(S.z**3)) * S.k

print(div(A))


def VecPot(xcomp, ycomp, zcomp):
    S = CoordSys3D('S')
    x, y, z = xcomp(S.x), ycomp(S.y), zcomp(S.y)
    return x * S.i + y * S.j + z * S.k


def Divergence(vec):
    return div(vec)
