import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


from snudd import config
from snudd.targets import Nucleus


from snudd.models  import GeneralNSI

from snudd.resolution import Resolution, Convolver # deals with the resolution effects on the rate

from snudd.resolution   import res_xnt_er

from snudd.efficiencies import efficiency_xnt_er_22 # efficiency curve from arxiv.org:2207.11330

from snudd import config
from snudd.targets import Electron
from snudd.binding import binding_xe
from snudd.rrpa    import rrpa_scaling

from scipy.optimize import minimize_scalar



# Background model B0 fit results: (central_value, error) table 1 arXiv:2207.11330
b0_fit = {
    "Pb214":            (960, 120),
    "Kr85":             (90, 60),
    "Materials":        (270, 50),
    "Xe136":            (1550, 50),
    "Xe124":            (250, 30),
    "AC":               (0.71, 0.03),
    "Xe133":            (150, 60),
    "Krm83m":           (80, 16),
}


def total_background(components: dict):
    values = np.array([v for v, _ in components.values()])
    errors = np.array([e for _, e in components.values()])
    total_val = values.sum()
    total_err = np.sqrt((errors**2).sum())  # quadrature sum
    return total_val, total_err


total_val, total_err = total_background(b0_fit)
rel_uncertainty = total_err / total_val

def counts_xnt(signal, E_Rs, EL, ER):
    exposure = 1.16 # in tn yr-1
    acceptance_cut = 0.995  # 91% is quoted in 2207.11330
    convolution_xnt_sig= Convolver(E_Rs, signal , efficiency_xnt_er_22, res_xnt_er)
    

    return convolution_xnt_sig.convolved_binned_rate(EL, ER)*exposure*acceptance_cut




##########################
# load XenonNT 2022 expected background with neutrino signal subtracted in the way we did previously (no rrpa, no binding)
##########################


ERs_mid, data_obs, XnT22_B0 = np.genfromtxt('../data/exps/xnt/2022_binned_data_bkgnonu.txt', unpack=True)





@np.vectorize
def loglike(Nsm, Nsig, Nbk):
    Nobs = Nsm + Nbk
    Nth = 1e-40 + Nsig + Nbk
    #print(Nobs, Nth)
    if Nobs > 50:
        return (Nobs*np.log(Nth) - Nth - Nobs*np.log(Nobs) + Nobs)
    else: 
        return (Nobs*np.log(Nth) - Nth - np.log(np.math.gamma(Nobs+1)))
    
def gauss(x, mu, sig):
    
    return ((1.0)/(np.sqrt(2*np.pi*sig**2)))*np.exp(-(x-mu)**2/(2*sig**2))



def plot():
    ELs = np.linspace(0.0, 58.0, 30)/1e6
    ERs = np.linspace(2.0, 60.0, 30)/1e6

    E_Rs = np.logspace(-2, 2, 1000) / 1e6  # E in GeV


    # Create host nucleus
    Xe_nucleus = Nucleus(54, 132, mass=131.9041535 * config.u) # single isotope 


    SM_matrix = np.array([[0, 0, 0],
                        [0, 0.0, 0],
                        [0, 0, 0.0]])

    eta = 0
    phi = 0

    sm_model = GeneralNSI(SM_matrix, eta, phi)


    # Create bound electron object
    Xe_electron = Electron(Xe_nucleus, binding_xe, rrpa_scaling) 
    Xe_electron.update_model(sm_model)
    Xe_electron.prepare_density()

    sm_spec_er = Xe_electron.spectrum(E_Rs)

    SM_counts = np.zeros(ERs.shape)
    for i in tqdm(range(len(ELs))):
        SM_counts[i] = counts_xnt(sm_spec_er, E_Rs, ELs[i], ERs[i])


        
    plt.bar((ERs[:30] + ELs[:30])/2 *1e6, XnT22_B0[:30] ,  width= 2.0, edgecolor='black', alpha=0.5, label=r'XnT22_B0', align='center')

    plt.bar((ELs+ERs)/2 *1e6, SM_counts, width=2.0, edgecolor='black', alpha=0.5, label=r'SM counts', align='center')


    data_err = np.sqrt(data_obs)

    # Plot with error bars
    plt.errorbar(ERs_mid[:30], data_obs[:30], 
                xerr=1.0, yerr=data_err[:30],
                fmt='o',              # marker style (like scatter)
                capsize=3,            # cap size on error bars
                capthick=1,           # cap thickness
                elinewidth=1,         # error bar line width
                markersize=5,
                label='XnT22 data',
                color='black')
    
    #plt.scatter(ERs_mid[:30], data_obs[:30], label=r'XnT22 data')
    plt.xlim(xmin=0.0,xmax=30.0)
    plt.show()

    # print(ERs_mid) 
    # print(XnT22_B0)
    
    #np.vectorize(counts_xnt)(sm_spec_er, E_Rs, ELs, ERs)



def poisson_chi2(N_nu: np.ndarray, N_bkg: np.ndarray, N_data: np.ndarray, alpha: float = 0.0, sigma_alpha: float = 1.0):
    """
    Compute the Poissonlikelihood-based chi-squared (Cash statistic).

    chi^2 = 2 * sum_i [ N_pred^i - N_data^i + N_data^i * ln(N_data^i / N_pred^i) ] + (alpha / sigma_alpha)^2

    Parameters
    ----------
    N_nu : np.ndarray
        Number of neutrino events in each bin (must be non-negative)
    N_bkg : np.ndarray
        Number of background events in each bin (must be non-negative)
    N_data : np.ndarray
        Observed number of events in each bin (must be non-negative)
    alpha : float, optional
        Nuisance parameter (default: 0.0)
    sigma_alpha : float, optional
        Uncertainty on the nuisance parameter (default: 1.0)

    Returns
    -------
    float
        The chi-squared value
    """
    N_pred = np.atleast_1d(N_nu + (1+alpha)*N_bkg)
    N_data = np.atleast_1d(N_data)

    assert len(N_pred) == len(N_data), "N_pred and N_data must have the same length"

    # Handle edge cases:
    # 1. When N_data = 0, the term N_data * ln(N_data / N_pred) -> 0 (limit as x->0 of x*ln x = 0)
    # 2. When N_pred = 0, the log term is problematic - skip these bins
    # 3. Use np.log to compute ln(N_data / N_pred) = ln(N_data) - ln(N_pred)

    valid_mask = N_pred > 0

    if not np.any(valid_mask):
        return np.inf  # all predictions are invalid

    N_pred_valid = N_pred[valid_mask]
    N_data_valid = N_data[valid_mask]

    # Compute term = N_pred - N_data + N_data * ln(N_data / N_pred)
    # For zero bins where N_data = 0, the ln term contributes zero
    term = N_pred_valid - N_data_valid + N_data_valid * (
        np.log(N_data_valid + 1e-300) - np.log(N_pred_valid)  # handles N_data = 0 naturally
    )

    # Sum over bins and multiply by 2
    chi2 = 2.0 * np.sum(term)

    # Add the nuisance parameter penalty term
    chi2 += (alpha / sigma_alpha)**2

    return chi2



def derivative_poisson_chi2(N_nu: np.ndarray, N_bkg: np.ndarray, N_data: np.ndarray, alpha: float = 0.0, sigma_alpha: float = 1.0):
    """
    Compute the derivative of the Poissonlikelihood-based chi-squared (Cash statistic).
    """
    N_nu   = np.atleast_1d(N_nu)
    N_bkg  = np.atleast_1d(N_bkg)
    N_data = np.atleast_1d(N_data)

    assert len(N_nu) == len(N_data), "N_nu and N_data must have the same length"
    assert len(N_bkg) == len(N_data), "N_nu and N_data must have the same length"

    # Handle edge cases:
    # 2. When N_pred = 0, the log term is problematic - skip these bins

    valid_mask = N_nu + (1+alpha)*N_bkg > 0

    if not np.any(valid_mask):
        return np.inf  # all predictions are invalid

    N_nu_valid = N_nu[valid_mask]
    N_bkg_valid = N_bkg[valid_mask]
    N_data_valid = N_data[valid_mask]

    # Compute term = N_pred - N_data + N_data * ln(N_data / N_pred)
    # For zero bins where N_data = 0, the ln term contributes zero
    term = N_bkg_valid - N_data_valid * N_bkg_valid / (N_nu_valid + N_bkg_valid * (1+alpha))

    # Sum over bins and multiply by 2
    deriv = 2.0 * np.sum(term) + 2 * alpha / sigma_alpha**2

    return deriv



from snudd.nsi import utils

def set_NSI_params(eps, eta, phi):
    
    eps_matrix = np.array([[0., 1.0, 0.], [1.0, 0.0, 0.], [0., 0, 0.0]]) * eps # eps_{e\mu} can change this structure 
    eps_matrix = utils.eps_matrix_sym(eps_matrix)
    nsi_model = GeneralNSI(eps_matrix, eta, phi)
    
    return nsi_model

def scan_no_nuisance_1D(eps_space):

    #plot()
    # Setting up grid 
    # eps_space = np.geomspace(1e-1, 0.5, 20)
    eta = 0.0 # therefore in the proton-neutron plane
    phi = np.pi/2 # therefore in the electron direction

    ELs = np.linspace(0.0, 58.0, 30)/1e6 ### Left of bin
    ERs = np.linspace(2.0, 60.0, 30)/1e6 ### Right of bin
    E_Rs = np.logspace(-2, 2, 1000) / 1e6  # E in GeV


    # Initialize a 2D array: shape (n_eps, n_bins)
    counts_all = np.zeros((len(eps_space), len(ELs)))

    for i, eps in enumerate(tqdm(eps_space, desc="Processing eps values")):
        NSI_eps_tmp = eps
        NSI_eta_tmp = eta  
        NSI_phi_tmp = phi  

        Xe_nucleus = Nucleus(54, 132, mass=131.9041535 * config.u) # single isotope 


        Xe_electron = Electron(Xe_nucleus, binding_xe, rrpa_scaling) 

        nsi_temp = set_NSI_params(NSI_eps_tmp, NSI_eta_tmp, NSI_phi_tmp)

        Xe_electron.update_model(nsi_temp)
        Xe_electron.prepare_density()

        signal_temp = Xe_electron.spectrum(E_Rs) 
        # Compute binned counts for this eps value
        counts_all[i, :] = np.array([counts_xnt(signal_temp, E_Rs, ELs[bc], ERs[bc]) for bc in range(len(ELs))])

    chi2_values = np.zeros(len(eps_space)) 

    for i in range(len(eps_space)):
        chi2_values[i] =   poisson_chi2(counts_all[i, :], XnT22_B0[:30],  data_obs[:30]) 

    
    return chi2_values

def scan_nuisance_1D(eps_space, eta, phi, rel_unc=0.05):
    from scipy.optimize import minimize_scalar, minimize

    #plot()
    # Setting up grid 
    #eps_space = np.geomspace(1e-1, 0.5, 20)


    ELs = np.linspace(0.0, 58.0, 30)/1e6 ### Left of bin
    ERs = np.linspace(2.0, 60.0, 30)/1e6 ### Right of bin
    E_Rs = np.logspace(-2, 2, 1000) / 1e6  # E in GeV


    # Initialize a 2D array: shape (n_eps, n_bins)
    counts_all = np.zeros((len(eps_space), len(ELs)))
    chi2_alpha_values = np.zeros(len(eps_space)) 

    for i, eps in enumerate(tqdm(eps_space, desc="Processing eps values")):
        NSI_eps_tmp = eps
        NSI_eta_tmp = eta  
        NSI_phi_tmp = phi  

        Xe_nucleus = Nucleus(54, 132, mass=131.9041535 * config.u) # single isotope 


        Xe_electron = Electron(Xe_nucleus, binding_xe, rrpa_scaling) 

        nsi_temp = set_NSI_params(NSI_eps_tmp, NSI_eta_tmp, NSI_phi_tmp)

        Xe_electron.update_model(nsi_temp)
        Xe_electron.prepare_density()

        signal_temp = Xe_electron.spectrum(E_Rs) 
        # Compute binned counts for this eps value
        counts_all[i, :] = np.array([counts_xnt(signal_temp, E_Rs, ELs[bc], ERs[bc]) for bc in range(len(ELs))])

        result = minimize(
            fun = lambda a: poisson_chi2(counts_all[i, :], XnT22_B0[:30], data_obs[:30], alpha=a, sigma_alpha=rel_unc), 
            x0=0,
            jac = lambda a: derivative_poisson_chi2(counts_all[i, :], XnT22_B0[:30], data_obs[:30], alpha=a, sigma_alpha=rel_unc), 
            # bounds=(-5, 5), 
            method="L-BFGS-B"
            )
        chi2_alpha_values[i] = result.fun

    Xe_nucleus = Nucleus(54, 132, mass=131.9041535 * config.u) # single isotope 


    SM_matrix = np.array([[0, 0, 0],
                        [0, 0.0, 0],
                        [0, 0, 0.0]])

    eta = 0
    phi = 0

    sm_model = GeneralNSI(SM_matrix, eta, phi)


    # Create bound electron object
    Xe_electron = Electron(Xe_nucleus, binding_xe, rrpa_scaling) 
    Xe_electron.update_model(sm_model)
    Xe_electron.prepare_density()

    sm_spec_er = Xe_electron.spectrum(E_Rs)

    SM_counts = [counts_xnt(sm_spec_er, E_Rs, ELs[i], ERs[i]) for i in range(len(ELs))]



    resultsm = minimize(
        fun = lambda a: poisson_chi2(SM_counts, XnT22_B0[:30], data_obs[:30], alpha=a, sigma_alpha=rel_unc), 
        x0=0,
        jac = lambda a: derivative_poisson_chi2(np.zeros_like(XnT22_B0[:30]), XnT22_B0[:30], data_obs[:30], alpha=a, sigma_alpha=rel_unc), 
        # bounds=(-5, 5), 
        method="L-BFGS-B"
        )
    
    chi2_alpha_sm = resultsm.fun
    
    return chi2_alpha_values, chi2_alpha_sm




if __name__ == "__main__":
    eps_space = np.geomspace(1e-1, 0.5, 20)
    eta = 0.0 # therefore in the proton-neutron plane
    phi = np.pi/2 # therefore in the electron direction

    chi2_values = scan_no_nuisance_1D(eps_space)
    print(rel_uncertainty)
    chi2_alpha_values, chi2_alpha_sm = scan_nuisance_1D(eps_space, eta, phi, rel_unc=rel_uncertainty)

    min_chi2 = min([min(chi2_alpha_values), chi2_alpha_sm])

    plt.semilogx(eps_space, chi2_values-np.min(chi2_values))
    plt.semilogx(eps_space, chi2_alpha_values-np.min(chi2_alpha_values))
    plt.semilogx(eps_space, chi2_alpha_values-min_chi2)

    plt.axhline(2.71, color='k', ls='--')

    plt.ylim(0,16)
    plt.xlabel(r'$\epsilon_{e\mu}$', size=14)
    plt.ylabel(r'$\chi^2 - \chi^2_{\rm min}$', size=14)

    plt.show()