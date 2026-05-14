"""Contains cross sections to be used in targets for any model you like."""
from __future__ import annotations

import typing
import numpy as np
from abc import ABC, abstractmethod


import snudd.config as config

if typing.TYPE_CHECKING:
    from snudd.targets import Nucleus, Electron








#-------------------------- Helper functions ----------------------------------







def _nuclear_prefactor(nucleus, E_R, E_nu):
    """Return commonly used nuclear model prefactor."""
    F_helm = nucleus.form_factor(E_R)
    return config.G_F ** 2 / np.pi * nucleus.mass * (
            1 - nucleus.mass * E_R / (2 * E_nu ** 2)) * F_helm ** 2



def _is_hermitian_3x3(eps_mat):
    """Checks whether the NSI matrix is a heritian 3x3 matrix."""

    A   = np.asarray(eps_mat)
    tol = 1e-10                # numerical tolerance

    # Check shape
    if A.shape != (3, 3):
        return False, "3x3"

    # Check hermiticity: A == A^\dagger 
    return np.allclose(A, A.conj().T, atol=tol), "hermitian"






#-------------------------- Model classes ----------------------------------






class Model(ABC):
    """Provides model interface."""

    @abstractmethod
    def nucleus_cross_section_flavour(self, nucleus: Nucleus, E_R, E_nu):
        """Return cross section for target by flavour. Energies in GeV."""
        pass

    @abstractmethod
    def electron_cross_section_flavour(self, electron: Electron, E_R, E_nu):
        """Return cross section for target by flavour. Energies in GeV."""
        pass





class GeneralNSI(Model):
    """A general NSI model, which takes a matrix of NSI couplings and angles."""

    def __init__(self, eps_matrix, eta, phi):
        # In general, eps_matrix is a 3x3 array of complex numbers (the NSI magnitudes)
        # Check if it is hermitian
        herm3x3, message = _is_hermitian_3x3(eps_matrix)
        if not herm3x3:
            print(
"""
        *** WARNING ***
NSI matrix 'eps_matrix' is not a {} matrix: 
{}
It should be a hermitian 3x3 matrix of the form:
[[eps_ee,   eps_em,   eps_et],
 [eps_em^*, eps_mm,   eps_mt],
 [eps_et^*, eps_mt^*, eps_tt]]

Code will run, but results may be unphysical!
""".format(message, eps_matrix)
            )
        self.eps_matrix = eps_matrix
        self.eta = eta
        self.phi = phi

    @property
    def xi_p(self):
        """Return proton-rotated charged part of the NSI factorisation."""
        return np.sqrt(5) * np.cos(self.eta) * np.cos(self.phi)

    @property
    def xi_n(self):
        """Return the neutron-rotated charged part of the NSI factorisation."""
        return np.sqrt(5) * np.sin(self.eta)

    @property
    def xi_u(self):
        """Return the up-quark-rotated charged part of the NSI factorisation."""
        return np.sqrt(5) / 3 * (2 * np.cos(self.eta) * np.cos(self.phi) - np.sin(self.eta))

    @property
    def xi_d(self):
        """Return the down-quark-rotated charged part of the NSI factorisation."""
        return np.sqrt(5) / 3 * (2 * np.sin(self.eta) - np.cos(self.eta) * np.cos(self.phi))

    @property
    def xi_e(self):
        """Return the electron-rotated charged part of the NSI factorisation."""
        return np.sqrt(5) * np.cos(self.eta) * np.sin(self.phi)

    def G_nucleus_coupling_matrix(self, nucleus):
        """Return the G coupling matrix."""

        return (self.xi_p * nucleus.Z + self.xi_n * nucleus.N) * self.eps_matrix


    def nucleus_cross_section_flavour(self, nucleus, E_R, E_nu):
        """Return flavour cross section matrix. Eneregy in GeV"""

        Q_nu_N   = nucleus.Q_nu_N
        G_matrix = self.G_nucleus_coupling_matrix(nucleus)

        cs_sm  = Q_nu_N ** 2 / 4 * np.diag((1,1,1))
        cs_int = - Q_nu_N * G_matrix.real
        cs_bsm = np.matmul(G_matrix, G_matrix.conjugate())

        return np.multiply.outer(_nuclear_prefactor(nucleus, E_R, E_nu),
                                 (cs_sm + cs_int + cs_bsm))


    def electron_cross_section_flavour(self, electron, E_R, E_nu):
        """Return flavour cross section matrix. Energies in GeV."""

        eps_matrix =  self.eps_matrix 
        xi_e = self.xi_e

        GL_matrix = (np.array([[1,0,0],[0,0,0],[0,0,0]])   
                    + config.g_L * np.diag([1,1,1]) 
                    + 0.5*eps_matrix*xi_e)

        GR_matrix = (config.g_R*np.diag([1,1,1]) 
                     + 0.5 * eps_matrix*xi_e)

        prefactor  = 2 * config.G_F ** 2 * config.m_e / np.pi

        Lterm_shape_enhancement = (E_R / E_nu).shape  # To get Lterm to be correct shape for sum

        Lterm  = np.multiply.outer(np.ones(Lterm_shape_enhancement), np.matmul(GL_matrix, GL_matrix.conjugate()))
        Rterm  = np.multiply.outer((1 - E_R/E_nu)**2, np.matmul(GR_matrix, GR_matrix.conjugate()))
        LRterm = np.multiply.outer(((config.m_e * E_R)/(2 * E_nu**2)), (np.matmul(GL_matrix, GR_matrix.conjugate())
                                    + np.matmul(GR_matrix, GL_matrix.conjugate())))

        return prefactor * (Lterm + Rterm - LRterm)

        




class SM(GeneralNSI):
    """Wrapper class for the Standard Model neutrino scattering behaviour."""

    def __init__(self):
        eps_zero = np.zeros((3,3))
        super().__init__(eps_zero, 0, 0)
