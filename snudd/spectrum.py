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


# Number of points to use in neutrino energy integration for continuous spectra
Enu_points = 250  


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

        E_nu_min = self.nu_minimum_energy(E_R)  # Minimum neutrino energies for given recoil energies, shape (N_E,)
        # Array of interpolating functions for density matrix elements as a function of neutrino energy for given source nu
        density_elements_flux = self.nu_density_elements[nu] 

        # MONOCHROMATIC neutrino sources (e.g. 7Be_3/8) have a delta-function flux
        if nu in config.NU_SOURCE_KEYS_MONO:
            E_nu_mono  = config.E_nus[nu][0] / 1000 # Mono energy in GeV, scalar
            E_nus_mins = (E_nu_min < E_nu_mono) # Check if mono energy is above minimum energy; if not, return zero
            dsigma_mat = self.target.cross_section_flavour(E_R, E_nu_mono) # Shape (N_E, 3, 3) 
            v_flux = np.array([[config.nu_flux[nu]]]) # Mono flux in per GeV, shape (1, 1) to allow broadcasting
            
            # integrated_cnadir_array = np.zeros(shape=(len(self.cnadirs), len(E_R)))
            # for icnadir in range(len(self.cnadirs)):
            #     density_mat = self.density_calc.matrix_from_elements(density_elements_flux[icnadir](E_nu_mono)) # Shape (3, 3)
            #     # matmul uses broadcasting on arrays of matrices, so we can input density_mat of shape (3, 3) and dsigma_mat of shape (N_E, 3, 3) 
            #     # and it will output an array of shape (N_E, 3, 3) where each (3, 3) matrix is the product of density_mat with the corresponding (3, 3) matrix in dsigma_mat.
            #     matrix_mult = np.matmul(density_mat, dsigma_mat) # Shape (N_E, 3, 3) after matmul, but we only have one E_nu, so effectively (3, 3)
            #     integrated = v_flux * matrix_mult.trace(axis1=-2, axis2=-1) * self.cnadir_weights[icnadir] # Shape (N_E, 1) after trace, but effectively scalar since we only have one E_nu
            #     integrated_cnadir_array[icnadir] = integrated
            # integrated_total = np.sum(integrated_cnadir_array, axis=0)
            # return self.target.number_targets_mass(E_R) * integrated_total * config.rate_conv * E_nus_mins
        

            density_eff = 0
            # Compute effective density matrix by summing over cnadir angles with corresponding weights
            for icnadir in range(len(self.cnadirs)):
                density_nad  = self.density_calc.matrix_from_elements(density_elements_flux[icnadir](E_nu_mono))  # Shape (3,3)
                density_eff += self.cnadir_weights[icnadir] * density_nad

            # ---- Perform matrix product and trace as einsum instead of matmul + trace ----
            # No integration needed for monoenergetic sources, so we just multiply flux by trace to get rate at mono energy, 
            # and then check if mono energy is above minimum energy to return zero if not.
            integrated = v_flux * np.einsum('ij,rij->r', density_eff, dsigma_mat)
            
            return  self.target.number_targets_mass(E_R) * integrated * config.rate_conv * E_nus_mins

        # # CONTINUOUS neutrino sources (e.g. 8B) 
        # nu_flux_fn = config.nu_flux_interp[nu] # Interpolating function for neutrino flux as a function of energy for given source nu, takes input in MeV and outputs in per MeV, so we will convert to GeV and per GeV when we use it below
        # # We need to check that the minimum neutrino energy is within the range of the interpolating function, otherwise we will get NaNs. 
        # # If it is outside the range, we can set it to the minimum or maximum energy of the interpolating function, 
        # # and then later we will multiply by zero if the mono energy is above the minimum energy to get zero rate as expected.
        # np.putmask(E_nu_min, E_nu_min < nu_flux_fn.x.min() / 1000, nu_flux_fn.x.min() / 1000)
        # np.putmask(E_nu_min, E_nu_min > nu_flux_fn.x.max() / 1000, (1 - 1e-6) * nu_flux_fn.x.max() / 1000)
        # E_nus = np.geomspace(E_nu_min, nu_flux_fn.x.max() / 1000, Enu_points)  # The relevant neutrino energies to integrate (in GeV)
        # nu_fluxes = nu_flux_fn(E_nus * 1000).T * 1e3  # Convert to per GeV
        # N_targets = self.target.number_targets_mass(E_R)

        # integrated_cnadir_array = np.zeros(shape=(len(self.cnadirs), len(E_R)))
        # E_R = np.array([E_R])
        # dsigma_mat = self.target.cross_section_flavour(E_R, E_nus)
        # dsigma_mat = dsigma_mat.swapaxes(0,1)
        # for icnadir in range(len(self.cnadirs)):
        #     density_mat = self.density_calc.matrix_from_elements(density_elements_flux[icnadir](E_nus))
        #     density_mat = np.rollaxis(density_mat, 3)
        #     matrix_mult = np.matmul(density_mat, dsigma_mat)
        #     matrix_mult = matrix_mult.swapaxes(0,1)

        #     integrands = nu_fluxes * matrix_mult.trace(axis1=-2, axis2=-1).T
        #     rates = N_targets * trapezoid(integrands, E_nus.T) * config.rate_conv * self.cnadir_weights[icnadir]
        #     integrated_cnadir_array[icnadir] = rates
        # integrated_total = np.sum(integrated_cnadir_array, axis=0)
        # return np.where(integrated_total < 0, 0, integrated_total)
    

        # CONTINUOUS neutrino sources (e.g. 8B) 
        nu_flux_fn = config.nu_flux_interp[nu]
        # We need to check that the minimum neutrino energy is within the range of the interpolating function, otherwise we will get NaNs. 
        # If it is outside the range, we can set it to the minimum or maximum energy of the interpolating function, 
        # and then later we will multiply by zero if the mono energy is above the minimum energy to get zero rate as expected.
        np.putmask(E_nu_min, E_nu_min < nu_flux_fn.x.min() / 1000, nu_flux_fn.x.min() / 1000)
        np.putmask(E_nu_min, E_nu_min > nu_flux_fn.x.max() / 1000, (1 - 1e-6) * nu_flux_fn.x.max() / 1000)
        # The relevant neutrino energies to integrate (in GeV)
        E_nus = np.geomspace(E_nu_min, nu_flux_fn.x.max() / 1000, Enu_points) # Shape (N_Enu, N_ER) 

        nu_fluxes = nu_flux_fn(E_nus * 1000).T * 1e3 # Convert to per GeV; shape (N_ER, N_Enu) after transpose

        E_R = np.array([E_R]) # Make E_R an array to allow broadcasting with E_nus

        # Compute cross section matrix for each E_R and E_nu; shape (N_Enu, N_ER, 3, 3)
        dsigma_mat = self.target.cross_section_flavour(E_R, E_nus)
        dsigma_mat = dsigma_mat.swapaxes(0, 1)  # (E_R, E_nu, 3, 3)

        # ---- Build weighted effective density ----
        density_all = []
        # Compute effective density matrix by summing over cnadir angles with corresponding weights
        for icnadir in range(len(self.cnadirs)):
            density_nad = self.density_calc.matrix_from_elements(density_elements_flux[icnadir](E_nus))  # Shape (E_nu, 3, 3, E_R) )
            density_nad = np.transpose(density_nad, (3, 0, 1, 2)) # (ER, Enu, 3, 3)
            density_all.append(density_nad * self.cnadir_weights[icnadir])

        density_all = np.stack(density_all)        # (Nc, E_R, E_nu, 3, 3)
        density_eff = np.sum(density_all, axis=0)  # (E_R, E_nu, 3, 3)

        # ---- Perform matrix product and trace as einsum instead of matmul + trace ----
        traces = np.einsum('reij,reji->re', density_eff, dsigma_mat, optimize=True) # Shape (E_R, E_nu) after einsum

        # ---- Single integration over averaged density ----
        integrands = nu_fluxes * traces # Shape (E_R, E_nu) after multiplication, ready for integration

        rates = self.target.number_targets_mass(E_R) * trapezoid(integrands, E_nus.T) * config.rate_conv
        rates[rates < 0] = 0. # Set any negative rates to zero, which can happen from numerical issues in the integration if the integrand is very small.

        return rates







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