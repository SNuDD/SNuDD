"""General nucleus and electron implementations."""
from abc import ABC, abstractmethod

import numpy as np

from snudd import config
from snudd.binding import ElectronBinder, binding_xe, binding_ar
from snudd.rrpa import rrpa_scaling
from snudd.spectrum import SpectrumTrace
from snudd.models import Model, SM
from snudd.nsi.oscillation import osc_params_best


class Target(ABC):
    """ABC for targets. Subclassed by Nucleus and Electron."""
    mass: float
    _spec: SpectrumTrace

    @abstractmethod
    def cross_section_flavour(self, E_R, E_nu):
        """Return flavour cross section."""
        pass

    @abstractmethod
    def number_targets_mass(self, E_R):
        """The number of targets per unit mass (generally a function of E_R)."""
        pass

    def momentum_transfer(self, E_R):
        """Return momentum transfer q in GeV,"""
        return np.sqrt(2 * self.mass * E_R)

    def update_model(self, model):
        self.model = model
        self._spec = SpectrumTrace(self)

    def update_oscillation_params(self, osc_params):
        self.osc_params = osc_params
        self._spec = SpectrumTrace(self)

    def spectrum(self, E_R, total=True, nu=None):
        """Return differential rate spectrum."""
        return self._spec.spectrum(E_R, total, nu)

    def prepare_probabilities(self):
        """Prepare probabilities for use in spectrum."""

        self._spec.prepare_probabilities()

    def prepare_density(self, cnadirs=[-1], cnadir_weights=[1], fast=False):
        """
        Prepare neutrino density matrix for use in spectrum.
        Give a list of cos(nadir) [cnadirs] and corresponding weights [cnadir_weights] to run earth matter evolution. 
        If cnadirs = [-1], no earth matter evolution is done and the density matrix is identity.
        Fast mode uses Magnus expansion for earth matter evolution instead of full numerical eigendecomposition.
        """
        if len(cnadirs) != len(cnadir_weights):
            raise ValueError("Cnadirs and cnadir_weights must have the same shape.")
        self._spec.prepare_density(cnadirs, cnadir_weights, fast)


class Nucleus(Target):
    """A nucleus and its associated electron. Can get momentum transfer, nuclear form factor, number or nuclei/mass,
    and number of free electrons/unit mass.
    """

    def __init__(self, Z, A, mass=None, model: Model = SM(), osc_params=osc_params_best):
        """Define nucleus by atomic number, weight, and mass. If mass not given, uses u*A."""
        self.Z = Z
        self.A = A
        self._mass = mass
        self.model = model
        self.osc_params = osc_params
        self._spec = SpectrumTrace(self)

    @property
    def mass(self):
        """Target mass."""
        return config.u * self.A if self._mass is None else self._mass

    @property
    def N(self):
        """Return the number of neutrons in the detector."""
        return self.A - self.Z

    @property
    def Q_nu_N(self):
        """Return the SM charge"""
        return self.N - (1 - 4 * config.sin_weak_2) * self.Z

    @property
    def E_max(self):
        """Return maximum recoil energy due to a neutrino collision (from kinematics)."""
        E_nu = config.E_nus['hep'][-1] / 1000  # Largest neutrino energy is from hep neutrinos (to GeV)
        return 2 * E_nu ** 2 / (self.mass + 2 * E_nu) * 1e6  # Conversion to keV

    def number_targets_mass(self, E_R=None):
        """Return the number of nuclei/unit mass of detector. kwarg for implementation with bound."""
        return 1 / self.mass

    def form_factor(self, E_R):
        """Return the form factor for the nucleus given recoil energy in GeV. Uses FormFactor component"""
        q = self.momentum_transfer(E_R) * config.fm_conv
        return helm_form_factor(q, self.A)

    def cross_section_flavour(self, E_R, E_nu):
        """Return flavour cross section matrix from given model."""
        return self.model.nucleus_cross_section_flavour(self, E_R, E_nu)


class Electron(Target):
    """An electron. Get momentum transfer and number free electrons/unit mass.
    """

    def __init__(self, nucleus, electron_binder: ElectronBinder, scaling=None, model: Model = SM()):
        """Electron belongs to a nucleus and is a component of it."""
        self.nucleus = nucleus
        self.electron_binder = electron_binder
        self.scaling = scaling if scaling is not None else lambda E: 1.
        self.osc_params = nucleus.osc_params
        self.model = model
        self._spec = SpectrumTrace(self)

    @property
    def mass(self):
        """Target mass."""
        return config.m_e

    def number_free_electrons(self, E_R):
        """Return the number of free electrons available to scatter per unit mass."""
        if self.electron_binder is not None:
            available_electrons = self.electron_binder.available_electrons(E_R)
        else:
            available_electrons = self.nucleus.Z

        return available_electrons * self.nucleus.number_targets_mass()

    def number_targets_mass(self, E_R):
        """Return scaled number of free electrons per unit mass. Use in RRPA."""
        return self.number_free_electrons(E_R) * self.scaling(E_R)

    def cross_section_flavour(self, E_R, E_nu):
        """Return flavour cross section from given model."""
        return self.model.electron_cross_section_flavour(self, E_R, E_nu)


def helm_form_factor(q, A):
    """Return Helm form factor presented in Lewin, J D and Smith, R F."""
    c = 1.23 * A ** (1 / 3) - 0.6  # fm
    s = 0.9  # fm
    a = 0.52  # fm
    R = np.sqrt(c ** 2 + 7 / 3 * np.pi ** 2 * a ** 2 - 5 * s ** 2)

    condlist = [q == 0., ]
    funclist = [1., lambda Q: 3 * j1(Q * R) / (Q * R) * np.exp(-Q ** 2 * s ** 2 / 2)]

    return np.piecewise(q, condlist, funclist)


def j1(x):
    """Return spherical Bessel function of the first kind."""
    try:
        bessel = (np.sin(x) - x * np.cos(x)) / x ** 2
    except FloatingPointError:
        bessel = 0.

    return bessel

nucleus_xe = Nucleus(config.Z_xe, config.A_xe, mass=config.mass_xe)
electron_xe = Electron(nucleus_xe, binding_xe, scaling=rrpa_scaling)

nucleus_ar = Nucleus(config.Z_ar, config.A_ar, mass=config.mass_ar)
electron_ar = Electron(nucleus_ar, binding_ar)