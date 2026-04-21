"""The solar probabilities module"""
import numpy as np
from scipy.interpolate import interp1d
from snudd import config
from snudd.nsi import flux_dists, oscillation as osc
from snudd import models


class ProbabilityCalculator:
    """The oscillation probabilities calculator."""

    def __init__(self, model, osc_params=osc.osc_params_best, adiabatic_check=False):
        self.model = model
        self.osc_params = osc_params
        self.adiabatic_check = adiabatic_check

    def prob_ee_3nu(self, E_nus, nu: str):
        "Return the electron survival probability in 3 nu picture."

        prob_2nu = self.prob_ee_2nu(E_nus, nu)

        result = self.osc_params.c13**4 * prob_2nu + self.osc_params.s13**4

        return result

    def prob_emu_3nu(self, E_nus, nu: str):
        "Return the electron to mu transition probability in 3 nu picture."

        prob_2nu = self.prob_ee_2nu(E_nus, nu)

        delta_term = 2 * self.osc_params.s13 * self.osc_params.s23 * self.osc_params.c23 * self.osc_params.s12 * self.osc_params.c12 * \
            np.cos(self.osc_params.delta_cp) * self._cos_matter_average(E_nus, nu)

        result = self.osc_params.c13**2 * (self.osc_params.c23**2 * (1 - prob_2nu) +
                               self.osc_params.s13**2 * self.osc_params.s23**2 * (1 + prob_2nu) +
                               delta_term)

        return result

    def prob_etau_3nu(self, E_nus, nu: str):
        "Return the electron to tau transition probability in 3 nu picture."

        prob_2nu = self.prob_ee_2nu(E_nus, nu)

        delta_term = 2 * self.osc_params.s13 * self.osc_params.s23 * self.osc_params.c23 * self.osc_params.s12 * self.osc_params.c12 * \
            np.cos(self.osc_params.delta_cp) * self._cos_matter_average(E_nus, nu)

        result = self.osc_params.c13**2 * (self.osc_params.s23**2 * (1 - prob_2nu) +
                               self.osc_params.s13**2 * self.osc_params.c23**2 * (1 + prob_2nu) -
                               delta_term)

        return result

    def prob_ee_2nu(self, E_nus, nu: str):
        "Return the electron survival probability in 2 nu picture."

        if self.adiabatic_check: osc.gamma_check(E_nus.max(), self.model, self.osc_params)

        return 0.5 * (1 + self._cos_matter_average(E_nus, nu) * self.osc_params.c12_2)

    def interpolate_probabilities(self, E_nu_min=3.4640e-3, E_nu_max=1.8784e1):
        """Return dictionary of interpolated probabilities for all nu sources.
        Interpolation done between neutrinos energies of E_nu_min and
        E_nu_max (MeV)
        """

        E_nus = np.geomspace(E_nu_min / 1e3, E_nu_max / 1e3, 500)  # GeV!

        interp_probabilities = {}
        for nu in config.NU_SOURCE_KEYS:
            probabilities = (self.prob_ee_3nu(E_nus, nu),
                             self.prob_emu_3nu(E_nus, nu),
                             self.prob_etau_3nu(E_nus, nu))
            interp_probabilities[nu] = interp1d(E_nus, probabilities)

        return interp_probabilities

    def _cos_matter_average(self, E_nus, nu: str):
        "Return the average of cos(2*theta_m)."

        xs = np.linspace(0., 0.35, 1000)  # Solar distances to integrate over
        integrand = osc.c12m_2(xs, E_nus, self.model, self.osc_params).T * \
            flux_dists.dist_dict[nu](xs)
        norm = config.trapezoid(flux_dists.dist_dict[nu](xs), xs)  # Account for slight lack of norm
        return config.trapezoid(integrand, xs) / norm









class DensityMatrixCalculator(ProbabilityCalculator):

    def __init__(self, model, osc_params=osc.osc_params_best, adiabatic_check=False):
        super().__init__(model, osc_params, adiabatic_check)


    # def delta_delta(self, cos_matter_averages):
    #     """CP dependent combination of mixing angles"""

    #     # PMNS mixing angles
    #     s13 = self.osc_params.s13
    #     s12, c12 = self.osc_params.s12, self.osc_params.c12
    #     s23, c23 = self.osc_params.s23, self.osc_params.c23
        
    #     # calculates the Delta_delta term
    #     d_delta = (0.5 * s13 * (2 * s12 * c12) * (2 * s23 * c23) * cos_matter_averages * np.cos(self.osc_params.delta_cp))
    #     return d_delta



    # def get_elements(self, E_nus, nu: str):
    #     """Calculate all individual elements of the rho density matrix"""
        
    #     # Shorthands for matter mixing quantities
    #     cosm_av  = self._cos_matter_average(E_nus, nu)
    #     p_ee_2nu = self.prob_ee_2nu(E_nus, nu)
    #     d_delta  = self.delta_delta(cosm_av)
        
    #     # PMNS mixing angles
    #     s13 = self.osc_params.s13
    #     c13 = self.osc_params.c13
    #     s13_2, c13_2 = s13**2, c13**2
    #     s23_2, c23_2 = self.osc_params.s23**2, self.osc_params.c23**2
        
    #     exp_delta = np.exp(1j * self.osc_params.delta_cp)
    #     sin_2theta12 = 2 * self.osc_params.s12 * self.osc_params.c12
    #     sin_2theta23 = 2 * self.osc_params.s23 * self.osc_params.c23
    #     cos_2theta23 = c23_2 - s23_2
        
    #     # diagonal density matrix elements
    #     r_ee = s13**4 + c13**4 * p_ee_2nu
    #     r_mm = c13_2 * (c23_2 * (1 - p_ee_2nu) + s13_2 * s23_2 * (1 + p_ee_2nu) + d_delta)
    #     r_tt = c13_2 * (s23_2 * (1 - p_ee_2nu) + s13_2 * c23_2 * (1 + p_ee_2nu) - d_delta)
        
    #     # off-diagonal elements 
    #     # electron-muon
    #     term_emu = (2 * s13 * self.osc_params.s23 * p_ee_2nu + 
    #                 self.osc_params.c23 * sin_2theta12 * cosm_av * exp_delta)
    #     r_em = c13 * s13**3 * self.osc_params.s23 - 0.5 * c13**3 * term_emu
        
    #     # electron-tau
    #     term_etau = (2 * s13 * self.osc_params.c23 * p_ee_2nu - 
    #                  self.osc_params.s23 * sin_2theta12 * cosm_av * exp_delta)
    #     r_et = c13 * s13**3 * self.osc_params.c23 - 0.5 * c13**3 * term_etau
        
    #     # muon-tau
    #     term1 = sin_2theta23 * ((1 + s13_2) * p_ee_2nu - c13_2)
    #     term2 = 2 * (cos_2theta23 / sin_2theta23) * d_delta
    #     imag_part = (-1j * np.sin(self.osc_params.delta_cp) * s13 * sin_2theta12 * cosm_av)
    #     r_mt = 0.5 * c13_2 * (term1 + term2 + imag_part)
        
    #     return r_ee, r_mm, r_tt, r_em, r_et, r_mt


    def density_mass(self, E_nus, nu: str):
        """Return the neutrino density matrix in the mass basis sampled at N energies of the energy array E_nus.
           Returns an array of matrices with shape (N, 3, 3).
        """

        # Shorthands for matter mixing quantities
        cos2m_av  = self._cos_matter_average(E_nus, nu) # Average of cos(2 theta)!!!!!
        cosm_avsq = (1+cos2m_av) / 2.   # <cos^2(theta)>
        s13       = self.osc_params.s13
        c13       = self.osc_params.c13

        # Ensure energies are passed as an array even for single energy
        n      = len(np.atleast_1d(E_nus))
        matrix = np.zeros((n, 3, 3), dtype=complex)
        
        # The three diagonal density matrix elements
        r_11  = c13**2 * cosm_avsq
        r_22  = c13**2 * (1-cosm_avsq)
        r_33  = s13**2 * np.ones_like(cosm_avsq)
        zeros = np.zeros_like(cosm_avsq)

        # Populate the density matrix
        matrix[:, 0, 0], matrix[:, 1, 1], matrix[:, 2, 2] = r_11, r_22, r_33
        matrix[:, 0, 1], matrix[:, 0, 2], matrix[:, 1, 2] = zeros, zeros, zeros 
        matrix[:, 1, 0], matrix[:, 2, 0], matrix[:, 2, 1] = zeros, zeros, zeros 
        
        return matrix 
    

    def density(self, E_nus, nu: str):
        """Return the neutrino density matrix in the flavour basis sampled at N energies of the energy array E_nus.
           Returns an array of matrices with shape (N, 3, 3).
        """

        # Define matrices
        density_mass = self.density_mass(E_nus, nu)
        matrix       = np.zeros_like(density_mass)

        # Convert to flavour space
        Upmns      = osc.UPMNS(self.osc_params)
        matrix[:]  = np.matmul(np.matmul(Upmns, density_mass[:]), Upmns.conj().T)

        return matrix












   
    def interpolate_density_elements(self, E_nu_min=3.4640e-3, E_nu_max=1.8784e1):

        """Return dictionary of interpolated de for all nu sources.
        Interpolation done between neutrinos energies of E_nu_min and
        E_nu_max (MeV)
        """

        E_nus = np.geomspace(E_nu_min / 1e3, E_nu_max / 1e3, 500)  # GeV!
        interp_expanded_rhos = {}

        for nu in config.NU_SOURCE_KEYS:
            rhos = self.density(E_nus, nu)
            rhoee = rhos[:, 0, 0]
            rhoemu = rhos[:, 0, 1]
            rhoeta = rhos[:, 0, 2]
            rhomumu = rhos[:, 1, 1]
            rhomuta = rhos[:, 1, 2]
            rhotata = rhos[:, 2, 2]

            expanded_rhos = (np.real(rhoee), np.imag(rhoee), 
                            np.real(rhoemu), np.imag(rhoemu),
                            np.real(rhoeta), np.imag(rhoeta),
                            np.real(rhomumu), np.imag(rhomumu),
                            np.real(rhomuta), np.imag(rhomuta),
                            np.real(rhotata), np.imag(rhotata))
            
            interp_expanded_rhos[nu] = interp1d(E_nus, expanded_rhos)

        return interp_expanded_rhos


    def matrix_from_elements(self, rho_els):
        ee_re, ee_im, emu_re, emu_im, eta_re, eta_im = (rho_els[0], rho_els[1], rho_els[2], rho_els[3],
                                                        rho_els[4], rho_els[5])
                                                    
        mumu_re, mumu_im, muta_re, muta_im, tata_re, tata_im= (rho_els[6], rho_els[7], rho_els[8], rho_els[9],
                                                                rho_els[10], rho_els[11])

        rho = np.array([[ee_re + 1j*ee_im, emu_re + 1j*emu_im, eta_re + 1j*eta_im],
                        [emu_re-1j*emu_im, mumu_re + 1j*mumu_im, muta_re+1j*muta_im],
                        [eta_re - 1j*eta_im, muta_re - 1j*muta_im, tata_re + 1j*tata_im]])

        if len(np.shape(rho)) > 2: 
            return rho.swapaxes(0, 2).swapaxes(1, 2)
        
    
        return rho


sm = models.GeneralNSI(np.zeros((3, 3)), 0, 0)
interp_density_sm = DensityMatrixCalculator(sm).interpolate_density_elements()
