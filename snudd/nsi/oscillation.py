"""Oscillation quantities."""

from dataclasses import dataclass
import numpy as np
from snudd import config
from snudd.nsi import solar_profiles


@dataclass
class OscillationParameters:
    """Dataclass holding the SM neutrino oscillation parameters"""

    delta_m12: float
    delta_m31: float 
    theta_12: float
    theta_13: float
    theta_23: float
    delta_cp: float

    @property
    def c12(self):
        """Cosine of theta_12 angle"""

        return np.cos(self.theta_12)

    @property
    def s12(self):
        """Sine of theta_12 angle"""

        return np.sin(self.theta_12)

    @property
    def c12_2(self):
        """Cosine of 2 * theta_12 angle"""

        return np.cos(2 * self.theta_12)

    @property
    def s12_2(self):
        """Sine of 2 * theta_12 angle"""

        return np.sin(2 * self.theta_12)

    @property
    def c13(self):
        """Cosine of theta_13 angle"""

        return np.cos(self.theta_13)
    @property
    def s13(self):
        """Sine of theta_13 angle"""

        return np.sin(self.theta_13)

    @property
    def c23(self):
        """Cosine of theta_23 angle"""

        return np.cos(self.theta_23)

    @property
    def s23(self):
        """Sine of theta_23 angle"""

        return np.sin(self.theta_23)

    @property
    def ordering(self) -> str:
        """Mass ordering: 'NO' if Δm²₃₁ > 0, else 'IO'."""
        return "NO" if self.delta_m31 > 0 else "IO"

    @property
    def delta_m32(self) -> float:
        """Compute Δm²₃₂ = Δm²₃₁ − Δm²₁₂."""
        return self.delta_m31 - self.delta_m12



def UPMNS(osc_params):
    """Return the PMNS mixing matrix."""

    c12 = osc_params.c12
    s12 = osc_params.s12
    c13 = osc_params.c13
    s13 = osc_params.s13
    c23 = osc_params.c23
    s23 = osc_params.s23
    delta_cp = osc_params.delta_cp


    U = np.array([[c12 * c13,
                   s12 * c13 * np.exp(1j * delta_cp),
                   s13],
                  [-np.exp(-1j * delta_cp) * s12 * c23 - c12 * s23 * s13,
                   c12 * c23 - s12 * s23 * s13 * np.exp(1j * delta_cp),
                   s23 * c13],
                  [np.exp(-1j * delta_cp) * s12 * s23 - c12 * c23 * s13,
                   -c12 * s23 - s12 * c23 * s13 * np.exp(1j * delta_cp),
                   c23 * c13]], dtype=np.complex128)

    return U



def eps_D(nsi_model, osc_params):
    """The eps_D parameter, found after performing the 3x3 -> 2x2 rotation."""

    # This is in general a 3x3 array of complex numbers
    eps_matrix = nsi_model.eps_matrix

    eps_ee     = eps_matrix[0][0]
    eps_mumu   = eps_matrix[1][1]
    eps_tautau = eps_matrix[2][2]
    eps_emu    = eps_matrix[0][1]
    eps_etau   = eps_matrix[0][2]
    eps_mutau  = eps_matrix[1][2]

    c13, s13, c23, s23 = osc_params.c13, osc_params.s13, osc_params.c23, osc_params.s23

    # Complex combination of NSI
    compln = s23 * eps_emu + c23 * eps_etau

    result = c13 * s13 * compln.real - \
             (1 + s13 ** 2) * c23 * s23 * eps_mutau.real - \
             0.5 * c13 ** 2 * (eps_ee - eps_mumu) + \
             0.5 * (s23 ** 2 - s13 ** 2 * c23 ** 2) * (eps_tautau - eps_mumu)

    return result


def eps_N(nsi_model, osc_params):
    """The eps_N parameter."""

    # This is in general a 3x3 array of complex numbers
    eps_matrix = nsi_model.eps_matrix

    eps_mumu   = eps_matrix[1][1]
    eps_tautau = eps_matrix[2][2]
    eps_emu    = eps_matrix[0][1]
    eps_etau   = eps_matrix[0][2]
    eps_mutau  = eps_matrix[1][2]

    c13, s13, c23, s23 = osc_params.c13, osc_params.s13, osc_params.c23, osc_params.s23

    result = c13 * (c23 * eps_emu - s23 * eps_etau) + \
             s13 * (s23 ** 2 * eps_mutau - c23 ** 2 * eps_mutau.conjugate() + c23 * s23 * (eps_tautau - eps_mumu))

    return result


def delta_vacuum_energy(E_nu, osc_params):
    """Return the difference in the vacuum energy eigenvalues between first and second mass eigenstates."""

    return osc_params.delta_m12 / (2 * E_nu)



"""----------------------The two fit-----------------------"""


def potential_cc(x):
    """Return charged-current potential (in GeV)."""

    return np.sqrt(2) * config.G_F * solar_profiles.electron_density(x)


def xi(x, nsi_model):
    """Return the total parameter xi given some nsi_model."""

    xi_charge = nsi_model.xi_p + nsi_model.xi_e

    return xi_charge + solar_profiles.neutron_electron_fraction(x) * nsi_model.xi_n


def potential_cc_dot(x):
    """Return derivative of charged-current potential with respect to solar fraction."""

    return np.sqrt(2) * config.G_F * solar_profiles.electron_density_derivative(x)


def xi_dot(x, nsi_model):
    """Return derivative of xi parameter with respect to solar fraction for a given NSI model."""

    return nsi_model.xi_n * solar_profiles.neutron_electron_fraction_derivative(x)





"""----------------------p and q-----------------------"""


def p(x, E_nu, nsi_model, osc_params):
    """Return our defined p parameter."""

    s12_2    = osc_params.s12_2
    delta_cp = osc_params.delta_cp
    d_vac    = delta_vacuum_energy(E_nu, osc_params) / 2 # Match the definition in Valeria's notes
    
    matter_ratio_real = np.multiply.outer(potential_cc(x) * xi(x, nsi_model) * eps_N(nsi_model, osc_params).real, 1/d_vac)
    matter_ratio_imag = np.multiply.outer(potential_cc(x) * xi(x, nsi_model) * eps_N(nsi_model, osc_params).imag, 1/d_vac)
    
    real_p = s12_2 * np.cos(delta_cp) + matter_ratio_real
    imag_p = s12_2 * np.sin(delta_cp) + matter_ratio_imag
    
    if delta_cp == 0:

        return np.squeeze(real_p)

    return np.squeeze(np.sign(real_p) * np.sqrt(real_p**2 + imag_p**2))


def q(x, E_nu, nsi_model, osc_params):
    """Return our defined q parameter."""

    c12_2 = osc_params.c12_2

    return np.squeeze(c12_2 + np.multiply.outer((xi(x, nsi_model) * eps_D(nsi_model, osc_params) - 0.5 * osc_params.c13 ** 2) *
                            potential_cc(x), 1 / (delta_vacuum_energy(E_nu, osc_params) / 2)))



def delta_matter_energy(x, E_nu, nsi_model, osc_params):
    """Return the difference in the matter energy eigenvalues between first and second mass eigenstates."""

    return delta_vacuum_energy(E_nu, osc_params) * np.sqrt(p(x, E_nu, nsi_model, osc_params) ** 2 + q(x, E_nu, nsi_model, osc_params) ** 2)




"""----------------------Angles-----------------------"""


def t12m_2(x, E_nu, nsi_model, osc_params):
    """Return the tangent of twice the mixing angle in matter."""

    return p(x, E_nu, nsi_model, osc_params) / q(x, E_nu, nsi_model, osc_params)


def s12m_2(x, E_nu, nsi_model, osc_params):
    """Return the sin of twice the mixing angle in matter."""

    return p(x, E_nu, nsi_model, osc_params) / (np.sqrt(p(x, E_nu, nsi_model, osc_params) ** 2 + q(x, E_nu, nsi_model, osc_params) ** 2))


def c12m_2(x, E_nu, nsi_model, osc_params):
    """Return the cos of twice the mixing angle in matter."""

    return q(x, E_nu, nsi_model, osc_params) / (np.sqrt(p(x, E_nu, nsi_model, osc_params) ** 2 + q(x, E_nu, nsi_model, osc_params) ** 2))





"""-------Here def of tanchi, tanchi_dot, chi_dot, theta_dot...------"""


def tanchi(x, E_nu, nsi_model, osc_params):
    """The tan of the effective matter mixing phase due to CP and/or complex NSI"""

    sin_2theta12 = osc_params.s12_2
    d_vac = delta_vacuum_energy(E_nu, osc_params) / 2

    epsN     = eps_N(nsi_model, osc_params)
    matt_vec = xi(x, nsi_model) * potential_cc(x)

    matter_real =  np.multiply.outer(matt_vec, epsN.real)
    matter_imag =  np.multiply.outer(matt_vec, epsN.imag)

    return - (d_vac * sin_2theta12 * np.sin(osc_params.delta_cp) + matter_imag) \
           / (d_vac * sin_2theta12 * np.cos(osc_params.delta_cp) + matter_real)


def f_dot(x, nsi_model):
    """Derivative of the combination f(x) = V(x) xi(x)"""

    return potential_cc_dot(x) * xi(x, nsi_model) + potential_cc(x) * xi_dot(x, nsi_model)


def tanchi_dot(x, E_nu, nsi_model, osc_params):

    d_vac    = delta_vacuum_energy(E_nu, osc_params) / 2
    epsN     = eps_N(nsi_model, osc_params)
    fdot     = f_dot(x, nsi_model)
    s12_2    = osc_params.s12_2
    delta_cp = osc_params.delta_cp

    matter_real = np.multiply.outer(xi(x, nsi_model) * potential_cc(x) * epsN.real, 1/d_vac)
    matter_imag = np.multiply.outer(xi(x, nsi_model) * potential_cc(x) * epsN.imag, 1/d_vac)

    A = s12_2*np.cos(delta_cp) + matter_real
    B = s12_2*np.sin(delta_cp) + matter_imag

    return fdot/(d_vac*A**2) * (B*epsN.real-A*epsN.imag)


def chi_dot(x, E_nu, nsi_model, osc_params):

    return np.cos(np.arctan(tanchi(x, E_nu, nsi_model, osc_params)))**2 * tanchi_dot(x, E_nu, nsi_model, osc_params)



def p_dot(x, E_nu, nsi_model, osc_params):
    """Derivative of the p parameter."""

    d_vac    = delta_vacuum_energy(E_nu, osc_params) / 2
    epsN     = eps_N(nsi_model, osc_params)
    fdot     = f_dot(x, nsi_model)
    s12_2    = osc_params.s12_2
    delta_cp = osc_params.delta_cp

    matter_real = np.multiply.outer(xi(x, nsi_model) * potential_cc(x) * epsN.real, 1/d_vac)
    matter_imag = np.multiply.outer(xi(x, nsi_model) * potential_cc(x) * epsN.imag, 1/d_vac)

    A = s12_2*np.cos(delta_cp) + matter_real
    B = s12_2*np.sin(delta_cp) + matter_imag

    if delta_cp == 0:
        
        return np.squeeze(2*epsN*fdot/d_vac)

    return fdot/delta_cp * (A*epsN.real+B*epsN.imag) / np.sqrt(A**2 + B**2)


def q_dot(x, E_nu, nsi_model, osc_params):
    """Derivative of the q parameter."""

    c13_2 = osc_params.c13 ** 2
    d_vac = delta_vacuum_energy(E_nu, osc_params)
    eps_D_val = eps_D(nsi_model, osc_params)

    return (2*f_dot(x, nsi_model) * eps_D_val - c13_2 * potential_cc_dot(x)) / d_vac



def theta_dot(x, E_nu, nsi_model, osc_params):
    """Derivative of the mixing angle in matter."""

    pval = p(x, E_nu, nsi_model, osc_params)
    qval = q(x, E_nu, nsi_model, osc_params)
    pdot = p_dot(x, E_nu, nsi_model, osc_params)
    qdot = q_dot(x, E_nu, nsi_model, osc_params)
    
    # Derivative of 0.5 * arctan(p/q)
    return 0.5 * (pdot * qval - pval * qdot) / (pval**2 + qval**2)




"""----------------------Gamma (Adiabaticity)-----------------------"""



def gamma(x, E_nu, nsi_model, osc_params):
    """Adiabaticty parameter in the Sun."""

    d_mat  = delta_matter_energy(x, E_nu, nsi_model, osc_params)    
    th_dot = theta_dot(x, E_nu, nsi_model, osc_params)
    ch_dot = chi_dot(x, E_nu, nsi_model, osc_params)
    c2m    = c12m_2(x, E_nu, nsi_model, osc_params)
    s2m    = s12m_2(x, E_nu, nsi_model, osc_params)
    
    den_plus = 1j * th_dot + 0.5 * s2m * ch_dot
    den_min  = 1j * th_dot - 0.5 * s2m * ch_dot
    den_max  = den_plus if np.all(np.abs(den_plus) >= np.abs(den_min)) else den_min
    
    return np.abs(d_mat - 0.5 * (1-c2m) * ch_dot) / (2 * np.abs(den_max)), "plus" if den_max is den_plus else "min"


def gamma_min(E_nu, nsi_model, osc_params):
    """Return the minimum value of gamma in the solar interior"""

    xs = np.linspace(0., 1., 100)
    gammas, choice = gamma(xs, E_nu, nsi_model, osc_params)
    return np.min(gammas), choice


def gamma_check(E_nu, nsi_model, osc_params, threshold=100):
    """Check if gamma value given (which should be a minimum value) is below
    a given threshold and warn the user if it is.
    """

    gamma_val = gamma_min(E_nu, nsi_model, osc_params)
    if gamma_val < threshold:
        print(f'Warning: minimum gamma is {gamma_val}, which is below set threshold of {threshold} for energy {E_nu} GeV. Adiabatic approximation may not be valid.')





"""----------------------Default oscillation parameters-----------------------"""




# OLD osc vals from 2006.11237
delta_m12_old = 7.50e-5 * (1e-9) ** 2  # GeV^2
delta_m31_old = 2.517e-3 *( 1e-9) ** 2 # GeV^2
theta_12_old  = 34.3 * np.pi / 180
theta_13_old  = 8.58 * np.pi / 180  # NORMAL ORDERING
theta_23_old  = 49.26 * np.pi / 180
delta_cp_old  = 0.0  # CP angle

osc_params_old = OscillationParameters(delta_m12_old,
                                        delta_m31_old,
                                        theta_12_old,
                                        theta_13_old,
                                        theta_23_old,
                                        delta_cp_old)


# UPDATED osc vals from 2410.05380 (IC24 w/ SK)
delta_m12 = 7.49e-5 * (1e-9) ** 2  # GeV^2
delta_m31 = 2.513e-3 *( 1e-9) ** 2
theta_12  = 33.68 * np.pi / 180
theta_13  = 8.56 * np.pi / 180  # NORMAL ORDERING
theta_23  = 43.3 * np.pi / 180
delta_cp  = 212  * np.pi / 180  # CP angle


osc_params_best = OscillationParameters(delta_m12,
                                        delta_m31,
                                        theta_12,
                                        theta_13,
                                        theta_23,
                                        delta_cp)
