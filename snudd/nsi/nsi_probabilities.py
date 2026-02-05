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
    """Class for computing the solar neutrino density matrix"""

    def __init__(self, model, osc_params=osc.osc_params_best, adiabatic_check=False):
        super().__init__(model, osc_params, adiabatic_check)


    @property
    def OMAT(self):

        return np.array([[self.osc_params.c13, 0, self.osc_params.s13],
                         [-self.osc_params.s13 * self.osc_params.s23, self.osc_params.c23, self.osc_params.c13 * self.osc_params.s23],
                         [-self.osc_params.s13 * self.osc_params.c23, -self.osc_params.s23, self.osc_params.c13 * self.osc_params.c23]])

    @property
    def AMAT(self):
        OMAT = self.OMAT
        return np.outer(OMAT[:, 1], OMAT[:, 1]) - np.outer(OMAT[:, 0], OMAT[:, 0])

    @property
    def BMAT(self):
        OMAT = self.OMAT
        return np.outer(OMAT[:, 0], OMAT[:, 1]) + np.outer(OMAT[:, 1], OMAT[:, 0])

    @property
    def CMAT(self):
        OMAT = self.OMAT
        return np.outer(OMAT[:, 0], OMAT[:, 1]) - np.outer(OMAT[:, 1], OMAT[:, 0])

    @property
    def DMAT(self):
        OMAT = self.OMAT
        return (np.outer(OMAT[:, 0], OMAT[:, 0]) * abs(OMAT[0, 0] * OMAT[0, 0])
                + np.outer(OMAT[:, 1], OMAT[:, 1]) * abs(OMAT[0, 1] * OMAT[0, 1])
                + np.outer(OMAT[:, 2], OMAT[:, 2]) * abs(OMAT[0, 2] * OMAT[0, 2]))




        

    
    def delta_delta(self, cos_matter_averages):
    # Delta_delta = 1/2 * s13 * sin(2*theta12) * sin(2*theta23) * cos(2*theta12_m) * cos(delta_CP)
    term = (0.5 * self.osc_params.s13 * (2 * self.osc_params.s12 * self.osc_params.c12) * (2 * self.osc_params.s23 * self.osc_params.c23) * cos_matter_averages * np.cos(self.osc_params.delta_cp))
    return term


    def prob_ee_2nu(self, E_nus, cos_matter_averages):
        "Return the electron survival probability in 2 nu picture."

        if self.adiabatic_check: osc.gamma_check(E_nus.max(), self.model, self.osc_params)

        return 0.5 * (1 + cos_matter_averages * self.osc_params.c12_2)


    def get_rho_matrix(self, E_nus, cos_matter_averages):
    p_ee_2nu = self.prob_ee_2nu(E_nus, cos_matter_averages)
    d_delta = self.delta_delta(cos_matter_averages)
    
    s13 = self.osc_params.s13
    c13 = self.osc_params.c13
    s13_2 = s13**2
    c13_2 = c13**2
    s23_2 = self.osc_params.s23**2
    c23_2 = self.osc_params.c23**2
    
    rho_ee = s13**4 + c13**4 * p_ee_2nu
    
    rho_mumu = c13_2 * (c23_2 * (1 - p_ee_2nu) + s13_2 * s23_2 * (1 + p_ee_2nu) + d_delta)
    
    rho_tautau = c13_2 * (s23_2 * (1 - p_ee_2nu) + s13_2 * c23_2 * (1 + p_ee_2nu) - d_delta)
    
    # complex terms (rho_emu, rho_etau, rho_mutau)

    exp_delta = np.exp(1j * self.osc_params.delta_cp)
    sin_2theta12 = 2 * self.osc_params.s12 * self.osc_params.c12
    
    term_emu = 2 * s13 * self.osc_params.s23 * p_ee_2nu + self.osc_params.c23 * sin_2theta12 * cos_matter_averages * exp_delta
    rho_emu = c13 * s13**3 * self.osc_params.s23 - 0.5 * c13**3 * term_emu
    
    term_etau = 2 * s13 * self.osc_params.c23 * p_ee_2nu - self.osc_params.s23 * sin_2theta12 * cos_matter_averages * exp_delta
    rho_etau = c13 * s13**3 * self.osc_params.c23 - 0.5 * c13**3 * term_etau

    return rho_ee, rho_mumu, rho_tautau, rho_emu, rho_etau


    def rho_mutau(self, cos_matter_averages):
    p_ee_2nu = self.prob_ee_2nu(None, cos_matter_averages) # E_nus non serve se hai già cos_matter
    d_delta = self.delta_delta(cos_matter_averages)
    
    s13 = self.osc_params.s13
    c13 = self.osc_params.c13
    s13_2 = s13**2
    c13_2 = c13**2
    
    sin_2theta12 = 2 * self.osc_params.s12 * self.osc_params.c12
    sin_2theta23 = 2 * self.osc_params.s23 * self.osc_params.c23
    cos_2theta23 = self.osc_params.c23**2 - self.osc_params.s23**2
    cot_2theta23 = cos_2theta23 / sin_2theta23
    
    # sin(2theta23) * ((1 + s13^2) * P_ee_2nu - c13^2)
    term1 = sin_2theta23 * ((1 + s13_2) * p_ee_2nu - c13_2)
    
    # 2 * cot(2theta23) * Delta_delta
    term2 = 2 * cot_2theta23 * d_delta
    
    # -i * sin(delta_CP) * s13 * sin(2theta12) * cos(2theta12_m)
    imaginary_part = -1j * np.sin(self.osc_params.delta_cp) * s13 * sin_2theta12 * cos_matter_averages
    
    rho_mutau_val = 0.5 * c13_2 * (term1 + term2 + imaginary_part)
    
    return rho_mutau_val










    def prob_2_osc(self, E_nus, cos_matter_averages):
        "Return the electron oscillation probability in 2 nu picture."
        return 1 - self.prob_ee_2nu(E_nus, cos_matter_averages)

    def prob_2_int(self, cos_matter_averages):
        "Return Prob_int = Re(S^(2)_11 (S^(2)_21)*) as defined in arXiv:2204.03011 eq. A10"
    
        c2m = cos_matter_averages
        return -self.osc_params.s12 * self.osc_params.c12 * c2m * np.cos(self.osc_params.delta_cp)

    def prob_2_ext(self, cos_matter_averages):
        "Return Prob_ext = Im(S^(2)_11 (S^(2)_21)*) as defined in arXiv:2204.03011 eq. A10"

        c2m = cos_matter_averages
        return self.osc_params.s12 * self.osc_params.c12 * c2m * np.sin(self.osc_params.delta_cp)

    def density(self,E_nus, nu: str):
        '''Equations A.17 from 2204.03011'''


        c2ms = self._cos_matter_average(E_nus, nu)


        DMAT_shape_enhancement = E_nus.shape  # To get Lterm to be correct shape for sum


        return self.osc_params.c13**2 * (np.multiply.outer(self.prob_2_osc(E_nus, c2ms), self.AMAT)
                        + np.multiply.outer(self.prob_2_int(c2ms), self.BMAT)
                        + 1j * np.multiply.outer(self.prob_2_ext(c2ms), self.CMAT)) + np.multiply.outer(np.ones(DMAT_shape_enhancement), self.DMAT)


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
