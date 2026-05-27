"""Provides the DD recoil differential rate spectrum."""
from __future__ import annotations
from tkinter import E

import typing

import numpy as np
from scipy.interpolate import interp1d
from snudd import config
from snudd.config import trapezoid
from snudd.nsi.nsi_probabilities import DensityMatrixEarthCalculator, interp_density_sm

if typing.TYPE_CHECKING:
    from snudd.targets import Target

class SpectrumTrace():
    """Target (nucleus or electron) spectrum."""

    def __init__(self, target):
        """Spectrum defined by target, model parameters, and probability type for neutrino probabilities."""
        self.target = target
        self.model = target.model
        self.osc_params = target.osc_params
        self.density_calc = DensityMatrixEarthCalculator(self.model, osc_params=self.osc_params, adiabatic_check=False)
        self.nu_density_elements = interp_density_sm
        self.cnadirs = [-1]
        self.cnadir_weights = [1]

    def nu_minimum_energy(self, E_R):
        """Return neutrino minimum energy given a recoil in GeV."""
        E_nu_min = 1. / 2. * (E_R + np.sqrt(E_R ** 2 + 2 * self.target.mass * E_R))

        return E_nu_min
    
    def _rate_nu(self, E_R, nu):
        """Return differential rate for a neutrino source. Overridden for each breakdown by subclasses."""

        E_nu_min = self.nu_minimum_energy(E_R)  # Minimum neutrino energy
        density_elements_flux = self.nu_density_elements[nu]
        if nu in config.NU_SOURCE_KEYS_MONO:
            E_nu_mono = config.E_nus[nu][0] / 1000
            E_nus_mins = (E_nu_min < E_nu_mono)
            dsigma_mat = self.target.cross_section_flavour(E_R, E_nu_mono)
            v_flux = np.array([[config.nu_flux[nu]]])
            integrated_cnadir_array = np.zeros(shape=(len(self.cnadirs), len(E_R)))
            for icnadir in range(len(self.cnadirs)):
                density_mat = self.density_calc.matrix_from_elements(density_elements_flux[icnadir](E_nu_mono))
                matrix_mult = np.matmul(density_mat, dsigma_mat)
                integrated = v_flux * matrix_mult.trace(axis1=-2, axis2=-1) * self.cnadir_weights[icnadir]
                integrated_cnadir_array[icnadir] = integrated
            integrated_total = np.sum(integrated_cnadir_array, axis=0)
            return self.target.number_targets_mass(E_R) * integrated_total * config.rate_conv * E_nus_mins

        nu_flux_fn = config.nu_flux_interp[nu]
        np.putmask(E_nu_min, E_nu_min < nu_flux_fn.x.min() / 1000, nu_flux_fn.x.min() / 1000)
        np.putmask(E_nu_min, E_nu_min > nu_flux_fn.x.max() / 1000, (1 - 1e-6) * nu_flux_fn.x.max() / 1000)
        E_nus = np.geomspace(E_nu_min, nu_flux_fn.x.max() / 1000, 500)  # The relevant neutrino energies (in GeV)
        nu_fluxes = nu_flux_fn(E_nus * 1000).T * 1e3  # Convert to per GeV
        N_targets = self.target.number_targets_mass(E_R)

        integrated_cnadir_array = np.zeros(shape=(len(self.cnadirs), len(E_R)))
        E_R = np.array([E_R])
        dsigma_mat = self.target.cross_section_flavour(E_R, E_nus)
        dsigma_mat = dsigma_mat.swapaxes(0,1)
        for icnadir in range(len(self.cnadirs)):
            density_mat = self.density_calc.matrix_from_elements(density_elements_flux[icnadir](E_nus))
            density_mat = np.rollaxis(density_mat, 3)
            matrix_mult = np.matmul(density_mat, dsigma_mat)
            matrix_mult = matrix_mult.swapaxes(0,1)

            integrands = nu_fluxes * matrix_mult.trace(axis1=-2, axis2=-1).T
            rates = N_targets * trapezoid(integrands, E_nus.T) * config.rate_conv * self.cnadir_weights[icnadir]
            integrated_cnadir_array[icnadir] = rates
        integrated_total = np.sum(integrated_cnadir_array, axis=0)
        return np.where(integrated_total < 0, 0, integrated_total)

    def prepare_density(self, cnadirs=[-1], cnadir_weights=[1]):
        """
        Return dictionary of interpolated probabilities for all nu sources.
        Interpolation done between neutrinos energies of E_nu_min and E_nu_max (MeV).
        Give a list of cos(nadir) [cnadirs] and corresponding weights [cnadir_weights] to run earth matter evolution. 
        If cnadirs = [-1], no earth matter evolution is done and the density matrix is identity.
        """
        if len(cnadirs) != len(cnadir_weights):
            raise ValueError("Cnadirs and cnadir_weights must have the same shape.")

        self.cnadirs             = cnadirs
        self.cnadir_weights      = cnadir_weights
        self.nu_density_elements = self.density_calc.interpolate_earth_density_elements(cnadirs)
        

    def spectrum(self, E_Rs, total=True, nu: str = None):
        """
        Return neutrino spectrum for ER in GeV, coupling g_x, A mass m_A, and neutrino type nu (if string).
        If nu an integer, returns sum over all neutrino spectra. If g_x = 0, we retrieve the SM spectrum (tested).
        """
        if nu is not None:
            spectrum_nu = self._spectrum_nu(E_Rs, nu)
            return spectrum_nu
        else:
            spectrum = np.array([self._spectrum_nu(E_Rs, key) for key in config.NU_SOURCE_KEYS])
            source_summed_spectrum = spectrum.sum(axis=0)
            return source_summed_spectrum

    def _spectrum_nu(self, E_Rs, nu):
        """Same as above but for specific source."""
        spectrum = self._rate_nu(E_Rs, nu)
        return np.squeeze(spectrum)

    def _total_spectrum(self, spectrum, total: bool):
        """Sum of zeroth axis of spectrum."""
        if total:
            return spectrum.sum(axis=0)
        return spectrum