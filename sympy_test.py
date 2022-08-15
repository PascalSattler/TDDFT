# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 16:43:37 2022

@author: Pascal Sattler
"""

from sympy import symbols, sympify, lambdify, log, pprint, Pow
import numpy as np
import matplotlib.pyplot as plt

A = 18.4029
B = 0.0
C = 7.50139
D = 0.101855
E = 0.01282710
alpha = 1.51124
beta = 0.2586
m = 4.42425

r_s = symbols("r_s")

logarithm = log(1 +alpha*r_s + beta*Pow(r_s,m))
fraction = (r_s + E*r_s**2)/(A + B*r_s + C*r_s**2 + D*r_s**3)

e_c = -0.5 * fraction * logarithm

V_cexp = - r_s * e_c.diff(r_s)+ e_c

V_c = lambdify(r_s, V_cexp, modules = ['numpy'])
xs = np.linspace(-49.95 , 49.95, 999)
plt.plot(xs, V_c(np.ones_like(xs)))
plt.show()
plt.close()