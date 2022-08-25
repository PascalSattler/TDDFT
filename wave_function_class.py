# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 14:16:33 2022

@author: Pascal Sattler
"""

import numpy as np
import matplotlib.pyplot as plt

class WaveFunction:
    
    def __init__(self, psi = None):
        if psi is not None:
            self.n_elec = psi.shape[1]
            self.size = psi.shape[0]
            self._create_psi(psi)
        
    def _create_psi(self, psi):
        self.psi = np.empty(2*self.n_elec*self.size)
        for i in range(self.n_elec):
            self.psi[2*i*self.size:(2*i+1)*self.size] = psi[:,i].real
            self.psi[(2*i+1)*self.size:(2*i+2)*self.size] = psi[:,i].imag
    
    def to_array(self):
        psi_array = np.reshape(self.psi, (self.n_elec, 2, self.size))
        return (psi_array[:,0] + 1j * psi_array[:,1])
    
    def probability(self, f_occ):
        rho = np.zeros(self.size)
        for i in range(self.n_elec):
            rho += f_occ[i] * (self.psi[2*i*self.size:(2*i+1)*self.size]**2 + self.psi[(2*i+1)*self.size:(2*i+2)*self.size]**2)
        return rho
    '''
    def dot(self, mat):
        if len(mat.shape) == 1:
           return self.dot_vec(mat)
        else:
            return self.dot_mat(mat)
        
    def dot_vec(self, vec):
        for i in range()
    '''
    
    def __mul__(self, mat):
        psi1 = self.to_array()
        psi2 = mat.to_array()
        return WaveFunction((psi1 * psi2).T)
    
    def copy(self):
        WF = WaveFunction()
        WF.psi = self.psi.copy()
        WF.size = self.size
        WF.n_elec = self.n_elec
        return WF
    
    def plot(self, x, f_occ):
        psi = self.to_array()
        rho = self.probability(f_occ)
        
        fig, ax = plt.subplot_mosaic([['left', 'right'],['bottom', 'bottom']])
        ax['left'].set_xlabel("x")
        ax['left'].set_ylabel("Re(Psi)")
        ax['left'].grid(True)
        ax['right'].set_xlabel("x")
        ax['right'].set_ylabel("Im(Psi)")
        ax['right'].grid(True)
        ax['bottom'].set_xlabel("x")
        ax['bottom'].set_ylabel("Rho")
        ax['bottom'].grid(True)
        
        for i in range(self.n_elec):
            ax['left'].plot(x, f_occ[i] * psi[i].real)
            ax['right'].plot(x, f_occ[i] * psi[i].imag)
            ax['bottom'].plot(x, rho)
            
        plt.tight_layout()
        plt.savefig("ground_state.pdf")
        plt.close()
        #plt.show()