"""Contains cross sections to be used in targets for any model you like."""
from __future__ import annotations

import typing
import numpy as np
from abc import ABC, abstractmethod


import snudd.config as config

if typing.TYPE_CHECKING:
    from snudd.targets import Nucleus, Electron








#-------------------------- Helper functions ----------------------------------





def _ER_max(Enu, mtarget):
    """Maximum recoil energy ERmax for neutrino energy Enu of target mith mass mtarget."""
    
    return 2 * Enu**2 / (mtarget + 2 * Enu)



def _nuclear_prefactor(nucleus, E_R, E_nu):
    """Return commonly used nuclear model prefactor."""
    
    F_helm = nucleus.form_factor(E_R)
    kin    = 1 - nucleus.mass * E_R / (2 * E_nu**2)
    pref   = config.G_F ** 2 / np.pi * nucleus.mass * kin * F_helm ** 2

    # Set cross section to zero if ER>ERmax
    ERmax = _ER_max(E_nu, nucleus.mass)
    mask  = E_R <= ERmax

    return pref * mask



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
    def electron_cross_section_flavour(self, E_R, E_nu):
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
        """Return flavour cross section matrix. Energy in GeV"""

        Q_nu_N   = nucleus.Q_nu_N
        G_matrix = self.G_nucleus_coupling_matrix(nucleus)

        # Define full amplitude
        # Note that we pull out a minus sign in the definition of Q_nu_N
        A   = -Q_nu_N/2 * np.eye(3) + G_matrix
        Asq = A.conjugate().T @ A

        return np.multiply.outer(_nuclear_prefactor(nucleus, E_R, E_nu), Asq)


    def electron_cross_section_flavour(self, E_R, E_nu):
        """Return flavour cross section matrix. Energies in GeV."""

        eps_matrix =  self.eps_matrix 
        xi_e = self.xi_e

        prefactor = 2 * config.G_F ** 2 * config.m_e / np.pi

        # LH and RH coupling matrices
        GL_matrix = np.diag([1,0,0]) + config.g_L * np.eye(3) + 0.5*eps_matrix*xi_e
        GR_matrix = config.g_R*np.eye(3) + 0.5 * eps_matrix*xi_e

        # Kinematic prefactors
        kinL  = np.ones_like(E_R / E_nu)  # To get Lterm to be correct shape for sum
        kinR  = (1 - E_R/E_nu)**2
        kinLR = (config.m_e * E_R)/(2 * E_nu**2)

        # Cross section terms
        Lterm  = np.multiply.outer(kinL,  GL_matrix.conjugate().T @ GL_matrix)
        Rterm  = np.multiply.outer(kinR,  GR_matrix.conjugate().T @ GR_matrix)
        LRterm = np.multiply.outer(kinLR, GL_matrix.conjugate().T @ GR_matrix + GR_matrix.conjugate().T @ GL_matrix)

        # Set cross section to zero if ER>ERmax
        ERmax = _ER_max(E_nu, config.m_e)
        mask  = E_R <= ERmax

        return prefactor * (Lterm + Rterm - LRterm) * mask[..., None, None]

        




class SM(GeneralNSI):
    """Wrapper class for the Standard Model neutrino scattering behaviour."""

    def __init__(self):
        eps_zero = np.zeros((3,3))
        super().__init__(eps_zero, 0, 0)
