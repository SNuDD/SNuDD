"""Oscillation quantities."""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import root_scalar
from snudd import config
from snudd.nsi import solar_profiles


@dataclass
class OscillationParameters:
    """Dataclass to holding oscillation parameters"""

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


def potential_cc(x):
    """Return charged-current potential (in GeV)."""

    return np.sqrt(2) * config.G_F * solar_profiles.electron_density(x)


def xi(x, nsi_model):
    """Return the total parameter xi given some nsi_model."""

    xi_charge = nsi_model.xi_p + nsi_model.xi_e

    return xi_charge + solar_profiles.neutron_electron_fraction(x) * nsi_model.xi_n



def eps_D(nsi_model, osc_params):
    """The eps_D parameter, found after performing the 3x3 -> 2x2 rotation."""

    eps_matrix = nsi_model.eps_matrix

    eps_ee = eps_matrix[0][0]
    eps_mumu = eps_matrix[1][1]
    eps_tautau = eps_matrix[2][2]
    eps_emu = eps_matrix[0][1]
    eps_etau = eps_matrix[0][2]
    eps_mutau = eps_matrix[1][2]

    c13, s13, c23, s23 = osc_params.c13, osc_params.s13, osc_params.c23, osc_params.s23

    result = c13 * s13 * (s23 * eps_emu + c23 * eps_etau) - \
             (1 + s13 ** 2) * c23 * s23 * eps_mutau - \
             0.5 * c13 ** 2 * (eps_ee - eps_mumu) + \
             0.5 * (s23 ** 2 - s13 ** 2 * c23 ** 2) * (eps_tautau - eps_mumu)

    return result


def eps_N(nsi_model, osc_params):
    """The eps_D parameter."""

    eps_matrix = nsi_model.eps_matrix

    eps_mumu = eps_matrix[1][1]
    eps_tautau = eps_matrix[2][2]
    eps_emu = eps_matrix[0][1]
    eps_etau = eps_matrix[0][2]
    eps_mutau = eps_matrix[1][2]

    c13, s13, c23, s23 = osc_params.c13, osc_params.s13, osc_params.c23, osc_params.s23

    result = c13 * (c23 * eps_emu - s23 * eps_etau) + \
             s13 * (s23 ** 2 * eps_mutau - c23 ** 2 * eps_mutau + c23 * s23 * (eps_tautau - eps_mumu))

    return result


def delta_vacuum_energy(E_nu, osc_params):
    """Return the difference in the vacuum energy eigenvalues between first and second mass eigenstates."""

    return osc_params.delta_m12 / (2 * E_nu)


def p(x, E_nu, nsi_model, osc_params):
    """Return our defined p parameter."""

    s12_2 = osc_params.s12_2

    return np.squeeze(s12_2 + np.multiply.outer(2 * xi(x, nsi_model) * eps_N(nsi_model, osc_params) *
                            potential_cc(x), 1 / delta_vacuum_energy(E_nu, osc_params)))


def q(x, E_nu, nsi_model, osc_params):
    """Return our defined p parameter."""

    c12_2 = osc_params.c12_2

    return np.squeeze(c12_2 + np.multiply.outer((2 * xi(x, nsi_model) * eps_D(nsi_model, osc_params) - osc_params.c13 ** 2) *
                            potential_cc(x), 1 / delta_vacuum_energy(E_nu, osc_params)))


def delta_matter_energy(x, E_nu, nsi_model, osc_params):
    """Return the difference in the matter energy eigenvalues between first
    and second mass eigenstates.
    """

    return delta_vacuum_energy(E_nu, osc_params) * np.sqrt(p(x, E_nu, nsi_model, osc_params) ** 2 +
                                                           q(x, E_nu, nsi_model, osc_params) ** 2)


def potential_cc_dot(x):
    """Return derivative of charged-current potential with respect to solar
    fraction.
    """

    return np.sqrt(2) * config.G_F * solar_profiles.electron_density_derivative(x)


def xi_dot(x, nsi_model):
    """Return derivative of xi parameter with respect to solar fraction for a given NSI model."""

    return nsi_model.xi_n * solar_profiles.neutron_electron_fraction_derivative(x)


def p_dot(x, E_nu, nsi_model, osc_params):
    """Return derivative of q parameter."""

    return np.multiply.outer(2 * eps_N(nsi_model, osc_params) *
                    (xi(x, nsi_model) * potential_cc_dot(x) + xi_dot(x, nsi_model) * potential_cc(x)),
                    1 / delta_vacuum_energy(E_nu, osc_params))


def q_dot(x, E_nu, nsi_model, osc_params):
    """Return derivative of q parameter."""

    c13 = osc_params.c13

    return np.multiply.outer((2 * eps_D(nsi_model, osc_params) * (xi(x, nsi_model) * potential_cc_dot(x) +
                                                                    xi_dot(x, nsi_model) * potential_cc(x)) -
                                            c13 ** 2 * potential_cc_dot(x)), 1 / delta_vacuum_energy(E_nu, osc_params))


def t12m_2(x, E_nu, nsi_model, osc_params):
    """Return the tangent of twice the mixing angle in matter."""

    return p(x, E_nu, nsi_model, osc_params) / q(x, E_nu, nsi_model, osc_params)


def s12m_2(x, E_nu, nsi_model, osc_params):
    """Return the sin of twice the mixing angle in matter."""

    return p(x, E_nu, nsi_model, osc_params) / (np.sqrt(p(x, E_nu, nsi_model,
                                                          osc_params) ** 2 + q(x, E_nu, nsi_model, osc_params) ** 2))


def c12m_2(x, E_nu, nsi_model, osc_params):
    """Return the cos of twice the mixing angle in matter."""

    return q(x, E_nu, nsi_model, osc_params) / (np.sqrt(p(x, E_nu, nsi_model, osc_params) ** 2 + q(x, E_nu, nsi_model, osc_params) ** 2))


def theta_dot(x, E_nu, nsi_model, osc_params):
    """Return the derivative of the mixing angle in matter."""

    result = 0.5 * (p_dot(x, E_nu, nsi_model, osc_params) * q(x, E_nu, nsi_model, osc_params) -
                    p(x, E_nu, nsi_model, osc_params) * q_dot(x, E_nu, nsi_model, osc_params)) / \
             (p(x, E_nu, nsi_model, osc_params) **2 + q(x, E_nu, nsi_model, osc_params) ** 2)

    return result


def gamma(x, E_nu, nsi_model, osc_params):
    """Return the general adiabaticity parameter."""

    return abs(delta_vacuum_energy(E_nu, osc_params) * (p(x, E_nu, nsi_model, osc_params) ** 2 + q(x, E_nu, nsi_model, osc_params) ** 2) ** 1.5 /
               (p_dot(x, E_nu, nsi_model, osc_params) * q(x, E_nu, nsi_model, osc_params) - p(x, E_nu, nsi_model, osc_params) * q_dot(x, E_nu, nsi_model, osc_params)))


def gamma_min(E_nu, nsi_model, osc_params):
    """Return the minimum value of gamma in the solar interior"""

    xs = np.linspace(0., 1., 100)
    gammas = gamma(xs, E_nu, nsi_model, osc_params)
    return np.min(gammas)


def gamma_check(E_nu, nsi_model, osc_params, threshold=100):
    """Check if gamma value given (which should be a minimum value) is below
    a given threshold and warn the user if it is.
    """

    gamma_val = gamma_min(E_nu, nsi_model, osc_params)
    if gamma_val < threshold:
        print(f'Warning: minimum gamma is {gamma_val}, which is below set threshold of {threshold} for energy {E_nu} GeV. Adiabatic approximation may not be valid.')



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
                   s12 * c13,
                   s13 * np.exp(-1j * delta_cp)],
                  [-s12 * c23 - c12 * s23 * s13 * np.exp(1j * delta_cp),
                   c12 * c23 - s12 * s23 * s13 * np.exp(1j * delta_cp),
                   s23 * c13],
                  [s12 * s23 - c12 * c23 * s13 * np.exp(1j * delta_cp),
                   -c12 * s23 - s12 * c23 * s13 * np.exp(1j * delta_cp),
                   c23 * c13]], dtype=np.complex128)

    return U



# OSC VALS FROM 2006.11237
delta_m12 = 7.50e-5 * (1e-9) ** 2  # GeV^2
delta_m31 = 2.517e-3 *( 1e-9) ** 2 # GeV^2
theta_12 = 34.3 * np.pi / 180
theta_13 = 8.58 * np.pi / 180  # NORMAL ORDERING
theta_23 = 49.26 * np.pi / 180
delta_cp = 0.0  # CP angle

osc_params_best = OscillationParameters(delta_m12,
                                        delta_m31,
                                        theta_12,
                                        theta_13,
                                        theta_23,
                                        delta_cp)
