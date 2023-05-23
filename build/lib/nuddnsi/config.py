import os
import numpy as np
import pandas as pd
from scipy.integrate import trapz
from scipy.interpolate import interp1d


_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data(path):
    return os.path.join(_ROOT, '../data', path)

# Constants
u = 931.49410242 / 1000  # Dalton in GeV
m_p = 938.27208816 / 1000  # Mass of proton in Gev
m_n = 939.56542052 / 1000  # Mass of neutron in GeV
m_e = 0.51099895000 / 1e3  # Mass of electron in GeV
m_mu = 105.66e-3  # 105.658375523 Mass of muon in GeV
m_tau = 1.77686  # Mass of tau in GeV
e = np.sqrt(4 * np.pi / 137)  # e in natural units
e0 = 1.60217662e-19  # e in Coulombs
c = 299792458  # c in m/s
G_F = 1.1663787e-5  # GF in GeV^-2
sin_weak_2 = 0.23122  # 0.2223  # sin squared of weak mixing angle
TONNE_YEAR = 60 * 60 * 24 * 365.25 * 1000  # Conversion from kg s to tonne yr
m_As_landscape = np.logspace(-8, 2, 150)
fm_conv = 5.0678047083  # Conversion from GeV to /fm
rate_conv = 1e-15 * 1.98e-14 ** 2 * c ** 2 / e0 * TONNE_YEAR

# Electron SM handedness factors
g_L = sin_weak_2 - 1 / 2
g_R = sin_weak_2

# Calculated constants
eps_mu_tau_prime = e / (6 * np.pi ** 2) * np.log(m_mu / m_tau)
L_mu_mass_scale_min = 1e2
L_mu_mass_scale_max = 1e16

# Detector constants
Z_ge = 32
A_ge = 73
mass_ge = 72.9234590 * u  # from ciaaw.org

Z_xe = 54
A_xe = 131
mass_xe = 130.90508414 * u  # from ciaaw.org

Z_ar = 18
A_ar = 40
mass_ar = 39.96238312 * u  # ciaaw.org

hyp_ge = {
    'E_threshs': [0.030, 0.010],
    'E_maxs': (None, 50),
    'exp': 0.56}  # Hypothetical Ge experiment

supercdms_hv_low = {
    'E_threshs': [0.040, 0.005],
    'E_maxs': (None, 7),
    'exp': 0.044}

supercdms_hv_high = {
    'E_threshs': [0.040, 0.005],
    'E_maxs': (2, 2),
    'exp': 0.044}

supercdms_low = {
    'E_threshs': [0.050, 0.020],
    'E_maxs': (None, 50),
    'exp': 0.056}

supercdms_high = {
    'E_threshs': [0.272, 0.120],
    'E_maxs': (None, 50),
    'exp': 0.056}

# inspired by arXiv:1802.06039v1
lz_low = {
    'E_threshs': [0.7, 0.7],
    'E_maxs': (None, 30),
    'exp': 15}  # 5.6 is fiducial mass. 1000 days is nominal live exposure

lz_high = {
    'E_threshs': [3., 2.],
    'E_maxs': (None, 30),
    'exp': 15}  # 5.6 is fiducial mass. 1000 days is nominal live exposure

# arXiv:1606.07001
darwin_low = {
    'E_threshs': [0.6, 0.6],  # Either 0.3 or 0.6 for NR
    'E_maxs': (None, 30),
    'exp': 200}  # 5.6 is fiducial mass. 1000 days is nominal live exposure

darwin_high = {
    'E_threshs': [3., 2.],
    'E_maxs': (None, 30),
    'exp': 200}

# arXiv:1707.08145 (Unsure about optimal)
darkside_low = {
    'E_threshs': [10, 2],
    'E_maxs': (None, 50),
    'exp': 100}

darkside_low_narrow = {
    'E_threshs': [0.6, 0.1],
    'E_maxs': (15.6, 3),
    'exp': 100}

darkside_high = {
    'E_threshs': [30, 7],
    'E_maxs': (None, 50),
    'exp': 100}

hyp_ar = {
    'E_threshs': [0.6, 0.1],
    'E_maxs': (None, 50),
    'exp': 100}  # Hypothetical argon experiment


# Ge Quenching fit constants
ge_quenching_consts = {'k': 0.1789,
                           'delta': 3.60,
                           'tau': 3.44,
                           'xi': 0.16,
                           'gamma': 1.367,
                           'x': 5.9}
# Paths
PATH_NU = get_data("nu_flux/")
PATH_PROB = get_data("v_probs/")  #get_data("data_old/nu_prob/")

# Dataframe
data_8B = np.loadtxt(PATH_NU + "B8_neutrino_flux.txt")  # 8B neutrino data (Bahcall 1995)
data_7Be_3 = np.loadtxt(PATH_NU + "7Be_3843_neutrino.txt")
data_7Be_8 = np.loadtxt(PATH_NU + "7Be_8613_neutrino.txt")
data_15O = np.loadtxt(PATH_NU + "15O_neutrino.txt")
data_17F = np.loadtxt(PATH_NU + "17F_neutrino.txt")
data_hep = np.loadtxt(PATH_NU + "hep_neutrino.txt")
data_13N = np.loadtxt(PATH_NU + "N13_neutrino.txt")
data_pp = np.loadtxt(PATH_NU + "pp_neutrino.txt")
data_atmos_mu = np.loadtxt(PATH_NU + "Atmos_mu_neutrino_flux.txt")  # Atmos v_mu flux at Super-K (Battistoni 2005)
data_atmos_e = np.loadtxt(PATH_NU + "Atmos_e_neutrino_flux.txt")  # Atmos v_e flux at Super-K (Battistoni 2005)
data_pep = np.loadtxt(PATH_NU + "pep_neutrino.txt")

df = pd.DataFrame({'Source': ['pp', '8B', 'hep', '7Be_3', '7Be_8', 'pep', '13N', '15O', '17F'],
                   'Energies': [data_pp[:, 0], data_8B[:, 0], data_hep[:, 0], data_7Be_3[:, 0] / 1000 + 0.3843,
                                data_7Be_8[:, 0] / 1000 + 0.8613, data_pep[:, 0], data_13N[:, 0], data_15O[:, 0], data_17F[:, 0]],
                   'Fluxes': [data_pp[:, 1], data_8B[:, 1], data_hep[:, 1], data_7Be_3[:, 1], data_7Be_8[:, 1],
                              data_pep[:, 1], data_13N[:, 1], data_15O[:, 1], data_17F[:, 1]],
                   'Total Flux': [5.98e10, 5.58e6, 8.04e3, 4.84e8, 4.35e9, 1.448e8, 2.97e8, 2.23e8, 5.52e6],
                   'Flux Unc': [0.006e10, 0.14e6, 1.30e3, 0.48e8, 0.35e9, 0.012e8, 0.14e8, 0.15e8, 0.17e6],
                   })

df = df.set_index(df.columns[0])

total_flux_elliot = {'8B': 5.58e6,  # From OLD GS98 HZ SSM
                     'pp': 5.98e10,
                     'hep': 8.04e3,
                     '7Be_3': 4.84e8,
                     '7Be_8': 4.35e9,
                     '13N': 2.97e8,
                     '15O': 2.23e8,
                     '17F': 5.52e6,
                     'pep': 1.448e8}

total_flux_GS98 = {  # New GS98 (HZ) (from 1611.09867)
                   'pp': 5.98e10,
                   '8B': 5.46e6,
                   'hep': 7.98e3,
                   '7Be_3': 0.1 * 4.93e9,
                   '7Be_8': 0.9 * 4.93e9,
                   'pep': 1.44e8,
                   '13N': 2.78e8,
                   '15O': 2.05e8,
                   '17F': 5.29e6,
                   }

total_flux_AGSS09met = {'8B': 4.50e6,  # New AGSS09met (LZ) (from above)
                        'pp': 6.03e10,
                        'hep': 8.25e3,
                        '7Be_3': 0.1 * 4.50e9,
                        '7Be_8': 0.9 * 4.50e9,
                        '13N': 2.04e8,
                        '15O': 1.44e8,
                        '17F': 3.26e6,
                        'pep': 1.46e8}

total_fluxes_ssm = {'original':total_flux_elliot, 'GS98':total_flux_GS98, 'AGSS09met': total_flux_AGSS09met}  # SSM
# Normalised fluxes
NU_SOURCE_KEYS = df.index
NU_SOURCE_INDS = {NU_SOURCE_KEYS[i]: i for i in range(len(NU_SOURCE_KEYS))}  # Mono is 3, 4, 8
NU_SOURCE_KEYS_MONO = ['7Be_3', '7Be_8', 'pep']  # Pick out monochromatic sources for special treatment
ssm_label = "GS98" # SSM label for total fluxes

integral = np.empty(len(NU_SOURCE_KEYS))
E_nus = df['Energies']
nu_fluxes = df['Fluxes']
nu_fluxes_uncs = df['Flux Unc']
nu_totals = total_fluxes_ssm[ssm_label]
nu_integral = {}
nu_flux = {}

for key in NU_SOURCE_KEYS:
    if key in NU_SOURCE_KEYS_MONO:
        continue
    else:
        nu_integral[key] = trapz(nu_fluxes[key], E_nus[key])

for key in NU_SOURCE_KEYS:
    if key in NU_SOURCE_KEYS_MONO:
        nu_flux[key] = nu_totals[key]
    else:
        nu_flux[key] = nu_fluxes[key] / nu_integral[key] * nu_totals[key]

nu_flux_interp = {key: interp1d(E_nus[key], nu_flux[key]) for key in
                  set(NU_SOURCE_KEYS).symmetric_difference(NU_SOURCE_KEYS_MONO)}

nu_flux_array = np.array(list((nu_flux.values())), dtype=object)  # np array of fluxes instead of dict
E_nus_array = np.array(list((E_nus.values)), dtype=object)
