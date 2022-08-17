# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 14:25:59 2022

@author: Pascal Sattler
"""

import numpy as np
from sympy import symbols, sympify, lambdify, exp, sin, cos, log, Pow, pprint, diff, zeros, simplify, Array, Matrix
from sympy.vector import CoordSys3D, express
from sympy.vector import divergence as div


class VectorPotential:
    
    Cart = CoordSys3D('Cart')
    Loc = Cart.create_new('Loc', transformation = lambda x, y, z: (x, y, z))
    
    def __init__(self, A_string, A_t = None, params = {}, transformation = None):
        vec_A = sympify(A_string)
        assert len(vec_A) == 3
        vec_A = vec_A[0] * Matrix([[1],[0],[0]]) + vec_A[1] * Matrix([[0],[1],[0]]) + vec_A[2] * Matrix([[0],[0],[1]])
        
        if transformation is not None:
            self.Loc = self.Cart.create_new('Loc', transformation = transformation)
        trafo = self.Loc.transformation_to_parent()
        self.symbols = self.Loc.base_scalars()
        self.jac = zeros(3, 3)
        for i in range(3):
            for j in range(3):
                self.jac[i, j] = simplify(diff(trafo[i], self.symbols[j]))
        self.inv_jac = self.jac.inv()
        
        for symbol in vec_A.free_symbols:
            if str(symbol) == 'x':
                vec_A = vec_A.subs({symbol : trafo[0]})
            elif str(symbol) == 'y':
                vec_A = vec_A.subs({symbol : trafo[1]})
            elif str(symbol) == 'z':
                vec_A = vec_A.subs({symbol : trafo[2]})
        
        vec_A = vec_A.subs({self.symbols[0] : 'q1', self.symbols[1] : 'q2', self.symbols[2] : 'q3'})
        self.inv_jac = self.inv_jac.subs({self.symbols[0] : 'q1', self.symbols[1] : 'q2', self.symbols[2] : 'q3'})
        
        self.vec_A = simplify(self.inv_jac * vec_A)
        self.vec_A = self.vec_A.subs(params)
        pprint(vec_A)
        
        self.A_t = A_t
        
    def get_A(self, t):
        t_dep = sympify(self.A_t)
        assert len(t_dep) == 3
        t_dep = t_dep[0] * Matrix([[1],[0],[0]]) + t_dep[1] * Matrix([[0],[1],[0]]) + t_dep[2] * Matrix([[0],[0],[1]])
        return self.vec_A * t_dep
    
    def get_div_A(self, t):
        pass
    
    def get_A_sqr(self, t):
        pass
    

    
VectorPotential('[x, y, z]', '[t, 1, t/2]', transformation = None) #lambda rho, phi, z : (rho * cos(phi), rho * sin(phi), z))

VectorPotential.get_A(t)