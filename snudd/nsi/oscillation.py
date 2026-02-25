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
    """The eps_N parameter."""

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

    return osc_params.delta_m12 / (4 * E_nu)



"""----------------------The two fit-----------------------"""

def ne_fit(x):

    N_A = 6.02214076e23
    u = (x / 0.075)**1.1
    exp_u = np.exp(-u)

    return N_A * 10**(2.36 - 4.52 * x - 0.33 * exp_u) * 7.619e-42 # Conversion cm^-3 -> GeV^3 (approx)


def ne_fit_der(x):

    ne = ne_fit(x)
    u = (x / 0.075)**1.1
    exp_u = np.exp(-u)
    ne_prime = -4.52 + 0.33 * (1.1 / 0.075) * np.power(x / 0.075, 0.1) * exp_u

    return ne * np.log(10) * ne_prime


def V_fit(x):

    ne = ne_fit(x)

    return np.sqrt(2) * config.G_F * ne


def V_fit_der(x):

    ne_d = ne_fit_der(x)

    return np.sqrt(2) * config.G_F * ne_d


def nn_fit(x):

    N_A = 6.02214076e23

    return N_A * 10**(1.72-4.80 * x) * 7.619e-42 # Conversion cm^-3 -> GeV^3 (approx)


def nn_fit_der(x):

    nn = nn_fit(x)

    return -4.8 * nn * np.log(10)


def xi_fit(x, nsi_model):

    ne = ne_fit(x)
    nn = nn_fit(x)

    xi_charge = nsi_model.xi_p + nsi_model.xi_e

    return xi_charge + nsi_model.xi_n * nn / ne


def xi_fit_der(x, nsi_model):

    ne = ne_fit(x)
    nn = nn_fit(x)
    ne_d = ne_fit_der(x)
    nn_d = nn_fit_der(x)

    return nsi_model.xi_n * (nn_d * ne - nn * ne_d / ne**2)




"""----------------------p and q-----------------------"""


def p(x, E_nu, nsi_model, osc_params):
    """Return our defined p parameter."""

    s12_2 = osc_params.s12_2
    delta_cp = osc_params.delta_cp
    d_vac = delta_vacuum_energy(E_nu, osc_params)
    
    matter_term = xi(x, nsi_model) * V_fit(x) * eps_N(nsi_model, osc_params)
    
    real_part = np.add.outer(matter_term, d_vac * s12_2 * np.cos(delta_cp))
    imag_part = np.add.outer(matter_term, d_vac * s12_2 * np.sin(delta_cp))
    
    return np.sqrt(real_part**2 + imag_part**2) / d_vac


def q(x, E_nu, nsi_model, osc_params):
    """Return our defined q parameter."""

    c12_2 = osc_params.c12_2
    c13_2 = osc_params.c13 ** 2
    d_vac = delta_vacuum_energy(E_nu, osc_params)
    
    matter_part = (xi(x, nsi_model) * eps_D(nsi_model, osc_params) - 0.5 * c13_2) * V_fit(x)
    
    return c12_2 + np.divide.outer(matter_part, d_vac)



def delta_matter_energy(x, E_nu, nsi_model, osc_params):
    """Return the difference in the matter energy eigenvalues between first and second mass eigenstates."""

    return delta_vacuum_energy(E_nu, osc_params) * np.sqrt(p(x, E_nu, nsi_model, osc_params) ** 2 + q(x, E_nu, nsi_model, osc_params) ** 2)


def potential_cc_dot(x):
    """Return derivative of charged-current potential with respect to solar fraction."""

    return np.sqrt(2) * config.G_F * solar_profiles.electron_density_derivative(x)


def xi_dot(x, nsi_model):
    """Return derivative of xi parameter with respect to solar fraction for a given NSI model."""

    return nsi_model.xi_n * solar_profiles.neutron_electron_fraction_derivative(x)




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

    sin_2theta12 = 2 * osc_params.s12 * osc_params.c12

    return - (delta_vacuum_energy(E_nu, osc_params) * sin_2theta12 * np.sin(osc_params.delta_cp)) / (delta_vacuum_energy(E_nu, osc_params) * 
                sin_2theta12 * np.cos(osc_params.delta_cp) + np.multiply.outer(xi(x, nsi_model) * V_fit(x), eps_N(nsi_model, osc_params)))



def f_dot(x, E_nu, nsi_model, osc_params):

    return V_fit_der(x) * xi_fit(x, nsi_model) + V_fit(x) * xi_fit_der(x, nsi_model)



def tanchi_dot(x, E_nu, nsi_model, osc_params):

    sin_2theta12 = 2 * osc_params.s12 * osc_params.c12

    return (delta_vacuum_energy(E_nu, osc_params) * sin_2theta12 * f_dot(x, E_nu, nsi_model, osc_params) *  np.sin(osc_params.delta_cp) * 
              eps_N(nsi_model, osc_params)) / (delta_vacuum_energy(E_nu, osc_params) * sin_2theta12 *  np.cos(osc_params.delta_cp) + 
              V_fit(x) * xi_fit(x, nsi_model) * eps_N(nsi_model, osc_params))**2


def chi_dot(x, E_nu, nsi_model, osc_params):

    return np.cos(np.arctan(tanchi(x, E_nu, nsi_model, osc_params)))**2 * tanchi_dot(x, E_nu, nsi_model, osc_params)



def p_dot(x, E_nu, nsi_model, osc_params):
    """Derivative of the p parameter with respect to solar radius x."""

    p_val = p(x, E_nu, nsi_model, osc_params)
    s12_2 = 2 * osc_params.s12 * osc_params.c12
    d_vac = delta_vacuum_energy(E_nu, osc_params)
    eps_N_val = eps_N(nsi_model, osc_params)
    f = V_fit(x) * xi_fit(x, nsi_model)
    fdot = f_dot(x, E_nu, nsi_model, osc_params)
    delta_cp = osc_params.delta_cp

    matter_term = np.multiply.outer(xi(x, nsi_model) * V_fit(x), eps_N(nsi_model, osc_params))
    
    real_part = np.add.outer(matter_term, d_vac * s12_2 * np.cos(delta_cp))
    imag_part = np.add.outer(matter_term, d_vac * s12_2 * np.sin(delta_cp))
    
    
    return (fdot * eps_N_val / d_vac) * (real_part**2 + imag_part**2)**(-1/2) * (real_part + imag_part)


def q_dot(x, E_nu, nsi_model, osc_params):
    """Derivative of the q parameter."""
    c13_2 = osc_params.c13 ** 2
    d_vac = delta_vacuum_energy(E_nu, osc_params)
    eps_D_val = eps_D(nsi_model, osc_params)

    return (f_dot(x, E_nu, nsi_model, osc_params) * eps_D_val - 0.5 * c13_2 * V_fit_der(x)) / d_vac



def theta_dot(x, E_nu, nsi_model, osc_params):
    """Derivative of the mixing angle in matter."""
    pval = p(x, E_nu, nsi_model, osc_params)
    qval = q(x, E_nu, nsi_model, osc_params)
    pdot = p_dot(x, E_nu, nsi_model, osc_params)
    qdot = q_dot(x, E_nu, nsi_model, osc_params)
    
    # Derivative of 0.5 * arctan(p/q)
    return 0.5 * (pdot * qval - pval * qdot) / (pval**2 + qval**2)




"""----------------------Gamma-----------------------"""


def gamma(x, E_nu, nsi_model, osc_params):
    """Adiabaticity parameter."""

    R_SUN_meters = 6.957e8
    R_SUN_GeV = R_SUN_meters * 5.06773e15

    d_vac = delta_vacuum_energy(E_nu, osc_params)
    pval = p(x, E_nu, nsi_model, osc_params)
    qval = q(x, E_nu, nsi_model, osc_params)
    d_mat = delta_matter_energy(x, E_nu, nsi_model, osc_params)    
    th_dot_x = theta_dot(x, E_nu, nsi_model, osc_params)
    ch_dot_x = chi_dot(x, E_nu, nsi_model, osc_params)
    s2m = s12m_2(x, E_nu, nsi_model, osc_params)
    c2m = c12m_2(x, E_nu, nsi_model, osc_params)

    th_dot = th_dot_x / R_SUN_GeV
    ch_dot = ch_dot_x / R_SUN_GeV
    
    den_plus = 1j * th_dot + 0.5 * s2m * ch_dot
    den_min =  1j * th_dot - 0.5 * s2m * ch_dot

    if np.all(np.abs(den_plus) >= np.abs(den_min)):
        den_max = den_plus
        choice = "plus"
    else:
        den_max = den_min
        choice = "min"
            
    return np.abs(0.5 * d_mat - 0.5 * (1-c2m) * ch_dot) / (np.abs(den_max)), choice



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
