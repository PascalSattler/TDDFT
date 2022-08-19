# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 12:18:00 2022

@author: Pascal Sattler
"""

import numpy as np
import scipy.integrate as integ
import matplotlib.pyplot as plt

def Phi(t, n_opt, w, E0):
    T = (2*np.pi/w)*n_opt
    val = np.piecewise(t, [t < 0, ((t >= 0) & (t <= T)), t > T], [0, lambda t : E0 * np.sin(w*t) * (np.sin(np.pi/T * t))**2, 0])
    return val

n_opt = 16
w = 2 * np.pi
E0 = 1
times = np.linspace(0, 1.1* (2*np.pi/w)*n_opt, 1000)

print(integ.trapezoid(Phi(times, n_opt, w, E0), times))

plt.grid(True)
plt.plot(times, Phi(times, n_opt, w, E0))