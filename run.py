# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 15:39:50 2022

@author: Pascal Sattler
"""
import numpy as np
import matplotlib.pyplot as plt
from DFT_classes import Hamiltonian, SoftCoulomb, TimePropagation
from wave_function_class import WaveFunction

L = 100
w = 0.05
xrange = [-L/2,L/2]
H = Hamiltonian(499, 2, xrange, fix = True)

def Vpot(x):
    return 0.5 * w**2 * (x)**2 #0.5 * w**2 * (x)**2 #np.cos(12*np.pi*x/L) #- SoftCoulomb(x, 0) #np.zeros_like(x)

H.create_ex_potential(Vpot)

H.solve_sc()

#H.plot_prob()

#H.solve()

momentum = WaveFunction(np.repeat(np.exp(1j * 1 * H.x).reshape(H.n, 1), H.Psi.n_elec, axis = -1))
ground_state = H.Psi * momentum
ground_state.plot(H.x, H.f_occ)

TP = TimePropagation(H, ground_state)

TP.time_prop(np.linspace(0, 200, 100))

TP.plot_prob()

TP.plot_dip()
