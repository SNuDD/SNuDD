# LZ scan 2025



import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocess import Pool
from pathlib import Path


# Loading SNuDD modules
from snudd import config
from snudd.targets import Electron, Nucleus
from snudd.binding import binding_xe
from snudd.rrpa import rrpa_scaling
from snudd.models import GeneralNSI, SM
from snudd.resolution import Resolution, Convolver, res_xnt_er, res_lz_nr
from snudd.efficiencies import efficiency_lz_b8
from snudd.nsi import utils





# Command line argument: ncores
#------------------------------------------------------------
# Set scan model and CPU s 
model = 'LZ_2025'

# Number of cores for parallelization
if len(sys.argv) > 1:
    ncores = int(sys.argv[1])
else:
    ncores = 8  # Default fallback if no argument is passed






# Global run parameters
#------------------------------------------------------------

# Recoil energy in GeV
E_Rs = np.logspace(-2, 2, 1000) / 1e6  

# ONE ISOTOPE: Xenon-132
Xe_nucleus = Nucleus(54, 132, mass=131.9041535 * config.u)






# Setting up paths
#------------------------------------------------------------

#respath = '/hydrarepo/vcosta/SNuDD/results'
respath = '/home/Valeria/Code/SNuDD/results'


outpath = os.path.join(respath, model)
Path(outpath).mkdir(parents=True, exist_ok=True)








# ## Functions
#------------------------------------------------------------------

def counts_LZB8_NR(signal, E_Rs):
    #### from https://arxiv.org/pdf/2512.08065 
    exposure = 5.7 # in tn yr-1
    convolution_lz_sig = Convolver(E_Rs, signal, efficiency_lz_b8, res_lz_nr)
    return convolution_lz_sig.convolved_binned_rate(1.0e-7, 1.5e-5) * exposure 

def set_NSI_params(eps, eta, phi):
    eps_matrix = np.array([[0., 1.0, 0.], [0., 0, 0.], [0., 0, 0.0]]) * eps 
    eps_matrix = utils.eps_matrix_sym(eps_matrix)
    nsi_model = GeneralNSI(eps_matrix, eta, phi)
    return nsi_model

def a_min(alpha_a, mu, n_obs):
    a = 1
    b = (1 + alpha_a**2 * mu)
    c = alpha_a**2 * (mu - n_obs)
    return (-b + np.sqrt(b**2 - 4*a*c)) / 2

def t_mu_a(alpha_a, mu, n_obs):
    """Return the t_mu statistic given an expected number of events and the SM expectation."""
    a = a_min(alpha_a, mu, n_obs)
    mu_adjusted = mu * (1 + a)
    return 2 * (mu_adjusted + n_obs*(np.log(n_obs / mu_adjusted) - 1)) + (a / alpha_a)**2







# ## SM Baseline Setup
#------------------------------------------------------------------
def init_sm_events():
    """Calculate the Standard Model expectation (observation) once."""
    sm_model = SM()
    Xe_nucleus.update_model(sm_model)
    Xe_nucleus.prepare_density()
    sm_spec_NR = Xe_nucleus.spectrum(E_Rs)
    return counts_LZB8_NR(np.array(sm_spec_NR).T, E_Rs)

# Pre-calculate SM expectation for the workers
N_LZ_sm = init_sm_events()





# ## Parallelizable worker function
#------------------------------------------------------------------
def calc_events(i, j, eps_val, phi_val):
    """Calculate the number of signal events in LZ for a specific grid point."""
    NSI_eta = 0.0
    NSI_model_tmp = set_NSI_params(eps_val, NSI_eta, phi_val)
    
    Xe_nucleus.update_model(NSI_model_tmp)
    Xe_nucleus.prepare_density()

    tmp_spec_NR = Xe_nucleus.spectrum(E_Rs)
    N_LZ_tmp = counts_LZB8_NR(np.array(tmp_spec_NR).T, E_Rs)

    # Return indices and the strictly real value to avoid ComplexWarnings later
    return i, j, np.real(N_LZ_tmp)









##########################################
#                  MAIN                  #
##########################################






def main():
    print(f"\nStarting {model} NSI scan using {ncores} cores...\n")

    # Construct the Grid
    phispace = np.linspace(-np.pi/2, np.pi/2, 8) 
    y_half = np.geomspace(0.01, 3.1, 8) 
    eps_space = np.concatenate([-y_half[::-1], y_half]) 

    phiGrid, epsGrid = np.meshgrid(phispace, eps_space, indexing='ij')

    N_LZ = np.zeros(np.shape(phiGrid))
    Tstat_LZ = np.zeros(np.shape(phiGrid))

    # Construct explicit argument list for starmap
    args = []
    for i in range(len(phispace)):
        for j in range(len(eps_space)):
            args.append((i, j, epsGrid[i,j], phiGrid[i,j]))

    # Run the parallel computation
    with Pool(processes=ncores) as pool:
        for result in tqdm(pool.starmap(calc_events, args), total=len(args), desc='Grid points'):
            i, j, N_LZ_tmp = result
            N_LZ[i, j] = N_LZ_tmp
            Tstat_LZ[i, j] = t_mu_a(0.12, N_LZ_tmp, N_LZ_sm) # 12% 8B uncertainty

    print("\nScan complete. Saving results and generating plots...")

    # Generate and save Test Statistic Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    
    plt.contourf(phiGrid, epsGrid, Tstat_LZ, levels=[6.18, np.inf], colors='tab:blue', alpha=0.3) 
    contLZ = plt.contour(phiGrid, epsGrid, Tstat_LZ, levels=[6.18], colors='tab:blue') 
    
    plt.clabel(contLZ, inline=1, fontsize=10)
    
    h1, l1 = contLZ.legend_elements()
    plt.legend([h1[0]], ["LZ 2025"])
    
    ax.set_yscale('symlog', linthresh=0.1, linscale=0.1)
    ax.set_ylim(-3, 3)
    plt.ylabel(r'$\varepsilon_{e\mu}$', size=14)
    plt.xlabel(r'$\varphi$', size=14)
    
    plot_file = os.path.join(outpath, "eps_emu_LL_LZ25.pdf")
    plt.savefig(plot_file, bbox_inches='tight')
    print(f"Plot saved to {plot_file}\n")





# Run main
if __name__ == "__main__":
    main()