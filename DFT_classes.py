# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 15:38:54 2022

@author: Pascal Sattler
"""

import numpy as np
from scipy import linalg as lin
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.ndimage import convolve as conv
from scipy.optimize import root
from scipy.integrate import ode, trapezoid
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.animation as anim
#from periodic_pulay_class import periodic_pulay
from wave_function_class import WaveFunction

def SoftCoulomb(x, x0):
    return 1/np.sqrt(1+(x - x0)**2)
    
def extrapolate(val, xrange, yrange):
    interp = interp1d(xrange, yrange, fill_value = 'extrapolate')
    return interp(val)

class Hamiltonian:
    
    Psi = None
    
    def __init__(self, n, n_elec, xrange, temp = 0, fix=True):
        self.n = n
        self.n_elec = n_elec
        assert hasattr(xrange, "__len__")
        assert len(xrange) == 2
        self.xmin = xrange[0]
        self.xmax = xrange[1]
        self.x_span = self.xmax - self.xmin
        self.fix = fix
        
        if fix:
            self.h = self.x_span/(n+1)
            self.x = np.linspace(self.xmin + self.h, self.xmax - self.h, self.n)
        else:
            self.h = self.x_span/n
            self.x = np.linspace(self.xmin, self.xmax, self.n, endpoint = False)
            
        if temp == 0:
            self.f_occ = np.ones(n_elec)
        else:
            raise NotImplementedError("noch nicht fertig für endliche Temp")
            
        self._create_kinetic()
        
        self.sc_range = np.arange(-n, n+ 1) * self.h
    
    def _create_kinetic(self):
        if self.fix:
            self.T = - (0.5 / self.h**2) * diags([1, -2, 1], [-1, 0, 1], shape=(self.n, self.n))
        else:
            self.T = - (0.5 / self.h**2) * diags([1,1, -2, 1,1], [1-self.n,-1, 0, 1,self.n-1], shape=(self.n, self.n))
    
    def create_ex_potential(self, func):
        self.V_ex = func(self.x)
        
    def solve(self, num_eig=None):
        if self.Psi is None:
            self.H = self.T + diags(self.V_ex)
        else:
            self.H = self.T + diags(self.V_ex + self.correlation_pot(0) + self.hartree_pot())
            
        if num_eig is None:
            self.E, self.Psi = eigsh(self.H, which = 'SA', k = self.n_elec)
        else:
            self.E, self.Psi = eigsh(self.H, which = 'SA', k = num_eig)
            
        self.Psi = WaveFunction(self.Psi/np.sqrt(self.h))
        self.prob_density = self.get_probability(self.Psi.to_array())
        
    def plot(self):
        print(self.E)
        psi = self.Psi.to_array()
        for i in range(self.Psi.n_elec):
            plt.plot(self.x, np.real(psi[:,i]))
        plt.close()
           
    def get_probability(self, Psi):
        return np.einsum('ji,j', np.abs(Psi)**2, self.f_occ)
    
    def get_probabilityReIm(self, Psi, size):
        return np.einsum('ji,j', Psi[:size]**2 + Psi[size:]**2, self.f_occ)
    
    def probability(self, psi):
        rho = np.zeros(self.n)
        for i in range(self.n_elec):
            rho += self.f_occ[i] * (psi[2*i*self.n:(2*i+1)*self.n]**2 + psi[(2*i+1)*self.n:(2*i+2)*self.n]**2)
        return rho
        
    def plot_prob(self):
        plt.plot(self.x, self.prob_density)
    
    def _get_wigner_seitz(self):
        self.r_s = np.abs(1/(2 * self.prob_density))
        
    def correlation_pot(self, pol):
        if pol == 0:
            A = 18.4029
            B = 0.0
            C = 7.50139
            D = 0.101855
            E = 0.01282710
            alpha = 1.51124
            beta = 0.2586
            exponent = 4.42425
        elif pol == 1:
            A = 5.2479
            B = 0.0
            C = 1.56823
            D = 0.1286150
            E = 0.0032074
            alpha = 0.053882
            beta = 1.56E-5
            exponent = 2.95899
        else:
            raise ValueError("Polarization must be 0 or 1!")
        
        self._get_wigner_seitz()
        
        fraction = (self.r_s + E*self.r_s**2)/(A + B*self.r_s + C*self.r_s**2 + D*self.r_s**3)
        logarithm = np.log(1 + alpha*self.r_s + beta*self.r_s**exponent)
        self.e_corr = -0.5 * fraction * logarithm
        
        fraction_derivative = (A + 2*A*E*self.r_s + (B*E-C)*self.r_s**2 - 2*D*self.r_s**3 - D*E*self.r_s**4 )/(A + B*self.r_s + C*self.r_s**2 + D*self.r_s**3)**2
        logarithm_derivative = (alpha + exponent*beta*self.r_s**(exponent-1))/(1 + alpha*self.r_s + beta*self.r_s**exponent)
        e_corr_derivative = self.r_s**2 * (fraction_derivative * logarithm + fraction * logarithm_derivative)
        
        return self.e_corr + self.prob_density * e_corr_derivative
    
    def hartree_pot(self):
        if self.fix:
            return self.h * conv(self.prob_density, SoftCoulomb(self.sc_range, 0), mode = 'constant', cval = 0)
        else:
            return self.h * conv(self.prob_density, SoftCoulomb(self.sc_range, 0), mode = 'wrap')
    ''' 
    def solve_sc(self):
        self.solve()
        n_diff = 1
        SC_scheme = periodic_pulay(len(self.prob_density))
        while n_diff > 1e-4:
            n_old = self.prob_density.copy()
            self.solve()
            f = self.prob_density - n_old 
            n_diff = lin.norm(f)
            self.prob_density = SC_scheme(n_old, f)
            print(n_diff)
    '''
    
    def solve_sc(self):
        if self.Psi is None:
            self.solve()
        N_it = root(self.iteration, self.prob_density, method = 'anderson', tol = 1e-6).nit
        print("{} iterations were performed for convergence.".format(N_it))

    def iteration(self, rho):
        self.prob_density = rho
        self.solve()
        return self.prob_density - rho
    


class TimePropagation:

    def __init__(self, hamiltonian: Hamiltonian , psi_start: np.ndarray):
        self.hamiltonian = hamiltonian
        self.psi_start = psi_start
        self.size = self.hamiltonian.n
        
    def _separateReImVec(self, vec):
        return np.concatenate((vec.real, vec.imag))
        
    def _get_H_time_t(self, t):
        self.H_R = self.hamiltonian.T
        self.V_ex = self.hamiltonian.V_ex
        self.V_xc = self.hamiltonian.correlation_pot(0)
        self.V_H = self.hamiltonian.hartree_pot()
        self.H_I = csr_matrix((self.size, self.size), dtype = np.float64)
    
    def psi_dt(self, t, psi):
        out = np.empty_like(psi)
        self._get_H_time_t(t)
        for i in range(self.psi_start.n_elec):
            psi_R = psi[2*i*self.size:(2*i+1)*self.size]
            psi_I = psi[(2*i+1)*self.size:(2*i+2)*self.size]
            out[2*i*self.size:(2*i+1)*self.size] =   self.H_R.dot(psi_I) + (self.V_ex + self.V_xc + self.V_H) * psi_I + self.H_I.dot(psi_R)
            out[(2*i+1)*self.size:(2*i+2)*self.size] = - self.H_R.dot(psi_R) - (self.V_ex + self.V_xc + self.V_H) * psi_R + self.H_I.dot(psi_I)
        return out
    
    def time_prop(self, times):
        self.times = times
        #print(1/(self.hamiltonian.E[-1] - self.hamiltonian.E[0]))
        prop = ode(self.psi_dt).set_integrator('dop853')#, first_step = 0.1)
        prop.set_initial_value(self.psi_start.psi.copy(), self.times[0])
        it = 1
        
        dip = dipole(self.hamiltonian.x, self.hamiltonian.fix)
        self.rho_list = np.empty((len(self.times), self.size), dtype = np.float64)
        self.rho_list[0] = self.psi_start.probability(self.hamiltonian.f_occ)
        
        self.dip_list = np.empty(len(self.times), dtype = np.complex128)
        self.dip_list[0] = dip.call(self.rho_list[0])
        
        while prop.successful() and prop.t < self.times[-1]:
            prop.integrate(self.times[it])
            self.rho_list[it] = self.hamiltonian.probability(prop.y)
            #psi_t = prop.y[:self.size] + 1j * prop.y[self.size:]
            
            self.dip_list[it] = dip.call(self.rho_list[it])
            print(prop.t, self.dip_list[it])
            
            #self.rho_extrap = extrapolate(self.times[it+1], self.times[:it], self.rho_list[:it])
            it += 1
    
    def _animate(self, i):
        self.line.set_ydata(self.rho_list[i])
        return self.line,
    
    def plot_prob(self):
        fig, ax = plt.subplots()
        xrange = self.hamiltonian.x
        ax.set_ylim(0, 1.1 * np.max(self.rho_list))
        self.line, = ax.plot(xrange, self.rho_list[0])
        animation = anim.FuncAnimation(fig, self._animate, interval = 20, blit = True, save_count = 50)
        animation.save("prob_density.gif", fps = 25)
        plt.show()
        
    def plot_dip(self):
        plt.title("Dipole moment")
        plt.xlabel("x")
        plt.grid(True)
        plt.plot(self.times, self.dip_list.real/self.hamiltonian.n_elec, label = "Re(dip)")
        plt.plot(self.times, self.dip_list.imag/self.hamiltonian.n_elec, label = "Im(dip)")
        plt.legend("lower right")
        plt.savefig("ground_state.pdf")
        plt.show()
    
class dipole:
    
    def __init__(self, x, fix = True):
        self.fix = fix
        if fix:
            dx_l = x[1] - x[0] 
            dx_r = x[-1] - x[-2]
            self.x = np.concatenate(([x[0] - dx_l], x, [x[-1] + dx_r]))
            self.call = self.dip_fix
        else:
            self.x = np.concatenate((x, [2*x[-1] - x[-2]]))
            self.L = self.x[-1] - self.x[0]
            self.call = self.dip_nfix
    
    def dip_fix(self, rho):
        integrand = np.concatenate(([0], rho, [0])) * self.x
        return trapezoid(integrand, self.x)
    
    def dip_nfix(self, rho):
        integrand = np.concatenate((rho, [rho[0]])) * np.exp(2j*np.pi*self.x/self.L)
        return trapezoid(integrand, self.x)
    
    
        
        
    
        
    
        

