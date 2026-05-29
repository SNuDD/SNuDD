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


# --- Backend selection (NumPy or Numba) ---
# Minor speedup from Numba for large scans
try:
    import numba as nb
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False




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
        self._cache_energy = {}     # caches E_nus + flux for each source to avoid repeated interpolation of the flux for each spectrum call, which can be expensive for large scans.
        self._cache_weights = {}    # cache trapezoidal weights for each unique E_nus grid to avoid repeated computation of trapezoidal weights for each spectrum call, which can be expensive for large scans.

    def nu_minimum_energy(self, E_R):
        """Return neutrino minimum energy given a recoil in GeV."""
        E_nu_min = 1. / 2. * (E_R + np.sqrt(E_R ** 2 + 2 * self.target.mass * E_R))

        return E_nu_min
    
    def _get_energy_cache(self, nu, E_R):
        """
        Cache (E_nus, flux) for given source and E_R.
        Saves computation time by avoiding repeated interpolation of the flux for each spectrum call.
        """
        key = (nu, hash(E_R.tobytes()))

        if key not in self._cache_energy:

            nu_flux_fn = config.nu_flux_interp[nu]
            E_nu_min = self.nu_minimum_energy(E_R)

            # Clip once here
            Emin = nu_flux_fn.x.min() / 1000
            Emax = nu_flux_fn.x.max() / 1000
            E_nu_min = np.clip(E_nu_min, Emin, (1 - 1e-6) * Emax)

            # Build grid ONLY ONCE
            E_nus = np.geomspace(E_nu_min, Emax, Enu_points)

            # Flux
            nu_fluxes = nu_flux_fn(E_nus * 1000).T * 1e3
            self._cache_energy[key] = (E_nus, nu_fluxes)

        return self._cache_energy[key]


    def _get_weights(self, E_nus):
        """Cache trapezoid weights for given E_nus to avoid repeated computation for each spectrum call.
        E_nus: shape (N_Enu, N_ER)"""
        key = hash(E_nus.tobytes())

        if key not in self._cache_weights:

            W = np.zeros_like(E_nus)

            for r in range(E_nus.shape[1]):
                W[:, r] = _trapezoid_weights(E_nus[:, r])

            self._cache_weights[key] = W

        return self._cache_weights[key]
      
    
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

            density_eff = 0
            # Compute effective density matrix by summing over cnadir angles with corresponding weights
            # density_eff: (3,3)
            for icnadir in range(len(self.cnadirs)):
                density_nad  = self.density_calc.matrix_from_elements(density_elements_flux[icnadir](E_nu_mono))  # Shape (3,3)
                density_eff += self.cnadir_weights[icnadir] * density_nad

            # ---- Perform matrix product and trace as einsum instead of matmul + trace ----
            # No integration needed for monoenergetic sources, so we just multiply flux by trace to get rate at mono energy, 
            # and then check if mono energy is above minimum energy to return zero if not.
            integrated = v_flux * np.einsum('ij,rij->r', density_eff, dsigma_mat) 
            return  self.target.number_targets_mass(E_R) * integrated * config.rate_conv * E_nus_mins    


        # CONTINUOUS neutrino sources (e.g. 8B) 
        # ---- Neutrino energy grid (in GeV) ----
        E_nus, nu_fluxes = self._get_energy_cache(nu, E_R)
        E_R = np.array([E_R]) # Make E_R an array to allow broadcasting with E_nus

        # Compute cross section matrix for each E_R and E_nu; shape (N_Enu, N_ER, 3, 3)
        dsigma_mat = self.target.cross_section_flavour(E_R, E_nus) # (E_nu, E_R, 3, 3)
        dsigma_mat = dsigma_mat.swapaxes(0, 1)  # (E_R, E_nu, 3, 3)

        # ---- Build weighted effective density ----
        density_eff = 0
        # Compute effective density matrix by summing over cnadir angles with corresponding weights
        for icnadir in range(len(self.cnadirs)):
            density_nad = self.density_calc.matrix_from_elements(density_elements_flux[icnadir](E_nus))  # Shape (E_nu, 3, 3, E_R) )
            density_eff += density_nad * self.cnadir_weights[icnadir]
        density_eff = np.transpose(density_eff, (3, 0, 1, 2)) # Get into correct shape (ER, Enu, 3, 3)


        if USE_NUMBA:
            # Use Numba-optimized kernel for speed if available, which performs the integration by hand to avoid overhead of np.einsum and np.trapz in the inner loop, which can be significant for large scans.
            weights_trap = self._get_weights(E_nus)   # cache this too!
            n_targets = self.target.number_targets_mass(E_R)
            rates = _rate_kernel_numba(nu_fluxes, density_eff, dsigma_mat, weights_trap)
            rates *= n_targets * config.rate_conv
        else:
            # Use straightforward NumPy implementation with np.einsum and scipy for clarity if Numba not available.
            # ---- Perform matrix product and trace as einsum instead of matmul + trace ----
            traces = np.einsum('reij,reji->re', density_eff, dsigma_mat) # Shape (E_R, E_nu) after einsum

            # ---- Single integration over averaged density ----
            integrands = nu_fluxes * traces # Shape (E_R, E_nu) after multiplication, ready for integration
            rates = self.target.number_targets_mass(E_R) * trapezoid(integrands, E_nus.T) * config.rate_conv

        rates[rates < 0] = 0. # Set any negative rates to zero, which can happen from numerical issues in the integration if the integrand is very small.
        return rates.real # Return real part of rates, as they should be real but can have small imaginary part from numerical issues in the integration.



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
            spectrum = np.stack([self._spectrum_nu(E_Rs, key) for key in config.NU_SOURCE_KEYS], axis=0)
            return spectrum.sum(axis=0)

    def _spectrum_nu(self, E_Rs, nu):
        """Same as above but for specific source."""
        spectrum = self._rate_nu(E_Rs, nu)
        return np.squeeze(spectrum)

    def _total_spectrum(self, spectrum, total: bool):
        """Sum of zeroth axis of spectrum."""
        if total:
            return spectrum.sum(axis=0)
        return spectrum
    












def _trapezoid_weights(x):
    """Return trapezoid weights for integration over x array."""
    w = np.zeros_like(x)

    dx = np.diff(x)

    w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    w[0] = dx[0] / 2
    w[-1] = dx[-1] / 2

    return w




@nb.njit(parallel=True, fastmath=True)
def _rate_kernel_numba(nu_fluxes, density_eff, dsigma_mat, weights):
    """
    Compute rates by explicit contraction and trapezoidal integration with precomputed weights, using Numba for speed. 
    This is much faster than np.einsum + np.trapz for large scans, as it avoids overhead of these functions in the inner loop.

    Shapes:
        nu_fluxes  : (N_ER, N_Enu)
        density_eff: (N_ER, N_Enu, 3, 3)
        dsigma_mat : (N_ER, N_Enu, 3, 3)
        weights    : (N_Enu, N_ER)  (trapezoidal weights for integration over E_nus, transposed to allow broadcasting with nu_fluxes)
    """

    N_ER, N_Enu = nu_fluxes.shape
    rates = np.zeros(N_ER, dtype=np.complex128)

    for r in nb.prange(N_ER):

        integral = 0.0

        for e in range(N_Enu):

            trace = 0.0
            for i in range(3):
                for j in range(3):
                    # Note the order of indices in density_eff and dsigma_mat for the trace: 
                    # we want sum_ij density_eff[r, e, i, j] * dsigma_mat[r, e, j, i]
                    trace += density_eff[r, e, i, j] * dsigma_mat[r, e, j, i]

            # Multiply by flux and trapezoidal weight for this E_nu and add to integral 
            # (handcrafting the integration here to avoid overhead of np.einsum and np.trapz in the inner loop, which can be significant for large scans)
            integral += nu_fluxes[r, e] * trace * weights[e, r]

        rates[r] = integral

    return rates



