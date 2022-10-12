

import numpy as np
from sympy import symbols, sympify, lambdify, exp, sin, cos, log, Pow, pprint, diff, zeros, simplify, Array, Matrix, Piecewise
from sympy.vector import CoordSys3D, express
from sympy.vector import divergence as div
from sympy.abc import t
import matplotlib.pyplot as plt

omega = 3
n_opt = 4
f0 = omega / (2 * np.pi)
pulse_time = (2*np.pi/omega)*n_opt
print(pulse_time)
E0 = 1
time = np.linspace(0, pulse_time, 8000)

#piece = 'piecewise(t, (sin(pi/T * t)**2, t < T/2), (sin(pi/T * t), t >= T/2))'
piece = 'piecewise(t, (t < T/2, sin(pi/T * t)**2), (t >= T/2, sin(pi/T * t)))'
Phi_t = '- E0 * sin(w*t) *' + piece # * sin(pi/T * t)**2'
params = {'w': omega, 'T': pulse_time, 'E0': E0}

Phi_t = sympify(Phi_t)
Phi_t = Phi_t.subs(params)
for symbol in Phi_t.free_symbols:
    if str(symbol) == 't':
        t = symbol
    else:
        raise ValueError('Phi_t has not defined parameters')
Phi_t = lambdify(t, Phi_t, modules = ['numpy'])
print(Phi_t(pulse_time / 2 - 1 / (f0 * 4)))

print(Phi_t(time))
plt.plot(time, Phi_t(time))
plt.grid(True)
plt.show()