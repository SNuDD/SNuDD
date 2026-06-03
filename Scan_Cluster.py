# LZ, XenonNT, PandaX scan cluster



import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocess import Pool
from pathlib import Path


# Loading SNuDD modules
from snudd import config
from snudd.targets import Electron, Nucleus
from snudd.geometry import SolarAngles
from snudd.models import GeneralNSI, SM
from snudd.resolution import Resolution, Convolver, res_xnt_nr, res_lz_nr, res_panda_nr
from snudd.efficiencies import efficiency_lz_b8, efficiency_pandaX, efficiency_xnt_2025
from snudd.nsi import utils






# Command line argument: ncores
#------------------------------------------------------------
# Set scan model and CPUs
model = 'Scan_AllExp_2025'

# 1st argument: Number of cores for parallelization
if len(sys.argv) > 1:
    ncores = int(sys.argv[1])
else:
    ncores = 8  # Default fallback

# 2nd argument: NSI element coefficient (1 to 6)
if len(sys.argv) > 2:
    nsi_elem = int(sys.argv[2])
else:
    nsi_elem = 4  # Default fallback (4 = emu)

print(f"Running with ncores = {ncores} and nsi_elem = {nsi_elem}")







# Global run parameters
#------------------------------------------------------------
# Recoil energy in GeV
E_Rs = np.logspace(-2, 2, 1000) / 1e6  


# NSI scan configuration
scan_phi = 0.0 # phi value for the scan in the nuclear plane


# Detector locations and solar angles
latitude_surf  = 44.35 # SURF (LZ)
latitude_gs    = 42.47 # Gran Sasso
latitude_panda = 28.15 # Jinping
t0 = 91     # Starting at perihelion (Jan 3rd)
T  = 182*2  # Data taking period


SURF      = SolarAngles(latitude=latitude_surf, t0=t0, T=T)
GranSasso = SolarAngles(latitude=latitude_gs, t0=t0, T=T)
Jinping   = SolarAngles(latitude=latitude_panda, t0=t0, T=T)


nbins = 30
cnadirs_surf, weights_surf = SURF.cnadir_hist(bins=nbins)
cnadirs_gran, weights_gran = GranSasso.cnadir_hist(bins=nbins)
cnadirs_jinping, weights_jinping = Jinping.cnadir_hist(bins=nbins)


# Isotope Data Setup (Filtered > 5% abundance)
Z_xe = 54
xe_iso_data = np.array([
    [129, 128.9047794 * config.u, 0.26401],
    [131, 130.9050824 * config.u, 0.21232],
    [132, 131.9041535 * config.u, 0.26909],
    [134, 133.9053945 * config.u, 0.10436],
    [136, 135.907219  * config.u, 0.08857]
]) 
# Renormalize the fractions to 1
xe_iso_data.T[2] /= xe_iso_data.T[2].sum()









# Setting up paths
#------------------------------------------------------------
# respath = '/hydrarepo/vcosta/SNuDD/results'
respath = '/home/Valeria/Code/SNuDD/results'

outpath = os.path.join(respath, model)
Path(outpath).mkdir(parents=True, exist_ok=True)









# ## Functions
#------------------------------------------------------------------

def unpack_isotope_data(isotope_data, Z):
    """Return array of nuclei for each isotope and array of isotopic fractions."""
    nuclei = []
    iso_fractions = []
    for (A_iso, mass_iso, iso_fraction) in isotope_data:
        nuclei.append(Nucleus(Z, A_iso, mass=mass_iso))
        iso_fractions.append(iso_fraction)
    return np.array(nuclei), np.array(iso_fractions)


# Create nuclei and isotopic fractions globally so workers can access them
nuclei, iso_fractions = unpack_isotope_data(xe_iso_data, Z_xe)


def average_isotope_spectra(nuclei, isotopic_fractions, E_Rs, model, cnadirs, weights):
    """Return weighted mean of energy spectra given nuclei and isotopic fractions."""
    spectra_iso = np.empty((len(nuclei), len(E_Rs)))
    for inucleus, nucleus in enumerate(nuclei):
        nucleus.update_model(model)
        if inucleus == 0:  
            nucleus.prepare_density(cnadirs=cnadirs, cnadir_weights=weights, fast=True)  
            prepared_density = nucleus._spec.nu_density_elements
        nucleus._spec.nu_density_elements = prepared_density
        spectrum_iso = nucleus.spectrum(E_Rs)
        spectra_iso[inucleus] = spectrum_iso
    return np.dot(isotopic_fractions, spectra_iso)


def counts_LZB8_NR(signal, E_Rs):
    exposure = 5.7 # in tn yr-1
    convolution_lz_sig = Convolver(E_Rs, signal, efficiency_lz_b8, res_lz_nr)
    return convolution_lz_sig.convolved_binned_rate(1.0e-7, 1.5e-5) * exposure 

def counts_XNT2025_NR(signal, E_Rs):
    exposure = 3.51 # in tn yr-1
    convolution_xnt_sig = Convolver(E_Rs, signal, efficiency_xnt_2025, res_xnt_nr)
    return convolution_xnt_sig.convolved_binned_rate(1.0e-7, 1.5e-5) * exposure 

def counts_PANDA_NR(signal, E_Rs):
    exposure = 1.20 # in tn yr-1
    convolution_panda_sig = Convolver(E_Rs, signal, efficiency_pandaX, res_panda_nr)
    return convolution_panda_sig.convolved_binned_rate(1.0e-7, 1.5e-5) * exposure


def set_NSI_params(eps, eta, phi, elem=1):
    """Return NSI model based on target element."""
    eps_matrix = np.zeros((3, 3), dtype=complex)
    if   elem == 1: eps_matrix[0, 0] = eps
    elif elem == 2: eps_matrix[1, 1] = eps
    elif elem == 3: eps_matrix[2, 2] = eps
    elif elem == 4: eps_matrix[0, 1] = eps
    elif elem == 5: eps_matrix[0, 2] = eps
    elif elem == 6: eps_matrix[1, 2] = eps
    else: 
        raise ValueError("Invalid element. Must be between 1 and 6.")
        
    eps_matrix = utils.eps_matrix_sym(eps_matrix)
    nsi_model = GeneralNSI(eps_matrix, eta, phi)
    return nsi_model


def a_min(alpha_a, mu, n_obs):
    a = 1
    b = (1 + alpha_a**2 * mu)
    c = alpha_a**2 * (mu - n_obs)
    return (-b + np.sqrt(b**2 - 4*a*c)) / 2

def t_mu_a(alpha_a, mu, n_obs):
    """Return the t_mu statistic with a Gaussian penalty."""
    a = a_min(alpha_a, mu, n_obs)
    mu_adjusted = mu * (1 + a)
    return 2 * (mu_adjusted + n_obs*(np.log(n_obs / mu_adjusted) - 1)) + (a / alpha_a)**2









# ## SM Baseline Setup
#------------------------------------------------------------------
def init_sm_events():
    """Calculate the Standard Model expectation for all 3 experiments once."""
    sm_model = SM()
    
    # Calculate LZ
    sm_spec_LZ = average_isotope_spectra(nuclei, iso_fractions, E_Rs, sm_model, cnadirs_surf, weights_surf)
    n_lz = counts_LZB8_NR(np.array(sm_spec_LZ).T, E_Rs)
    
    # Calculate XenonNT
    sm_spec_XNT = average_isotope_spectra(nuclei, iso_fractions, E_Rs, sm_model, cnadirs_gran, weights_gran)
    n_xnt = counts_XNT2025_NR(np.array(sm_spec_XNT).T, E_Rs)
    
    # Calculate PandaX
    sm_spec_PANDA = average_isotope_spectra(nuclei, iso_fractions, E_Rs, sm_model, cnadirs_jinping, weights_jinping)
    n_panda = counts_PANDA_NR(np.array(sm_spec_PANDA).T, E_Rs)
    
    return np.real(n_lz), np.real(n_xnt), np.real(n_panda)

# Pre-calculate SM expectation for the workers
N_LZ_sm, N_XNT_sm, N_PANDA_sm = init_sm_events()







# ## Parallelizable worker function
#------------------------------------------------------------------
def calc_events(i, j, eps_val, eta_val):
    """Calculate the number of signal events for all 3 experiments for a specific grid point."""
    NSI_model_tmp = set_NSI_params(eps_val, eta_val, scan_phi, elem=nsi_elem)
    
    # Prepare isotopic spectra
    tmp_spec_LZ = average_isotope_spectra(nuclei, iso_fractions, E_Rs, NSI_model_tmp, cnadirs_surf, weights_surf)
    tmp_spec_XNT = average_isotope_spectra(nuclei, iso_fractions, E_Rs, NSI_model_tmp, cnadirs_gran, weights_gran)
    tmp_spec_PANDA = average_isotope_spectra(nuclei, iso_fractions, E_Rs, NSI_model_tmp, cnadirs_jinping, weights_jinping)

    # Compute convolved events
    N_LZ_tmp = counts_LZB8_NR(np.array(tmp_spec_LZ).T, E_Rs)
    N_XNT_tmp = counts_XNT2025_NR(np.array(tmp_spec_XNT).T, E_Rs)
    N_PANDA_tmp = counts_PANDA_NR(np.array(tmp_spec_PANDA).T, E_Rs)

    # Return indices and the strictly real values
    return i, j, np.real(N_LZ_tmp), np.real(N_XNT_tmp), np.real(N_PANDA_tmp)








##########################################
#                  MAIN                  #
##########################################

def main():
    print(f"\nStarting {model} NSI scan using {ncores} cores...\n")

    # Construct the Grid
    etaspace = np.linspace(-np.pi/2, np.pi/2, 5) 
    y_half = np.geomspace(0.01, 3.1, 5) 
    eps_space = np.concatenate([-y_half[::-1], y_half]) 

    etaGrid, epsGrid = np.meshgrid(etaspace, eps_space, indexing='ij')

    # Initialize results containers
    N_LZ, Tstat_LZ       = np.zeros(np.shape(etaGrid)), np.zeros(np.shape(etaGrid))
    N_XNT, Tstat_XNT     = np.zeros(np.shape(etaGrid)), np.zeros(np.shape(etaGrid))
    N_PANDA, Tstat_PANDA = np.zeros(np.shape(etaGrid)), np.zeros(np.shape(etaGrid))

    # Construct explicit argument list for starmap
    args = []
    for i in range(len(etaspace)):
        for j in range(len(eps_space)):
            args.append((i, j, epsGrid[i,j], etaGrid[i,j]))

    # Run the parallel computation
    with Pool(processes=ncores) as pool:
        for result in tqdm(pool.starmap(calc_events, args), total=len(args), desc='Grid points'):
            i, j, N_LZ_tmp, N_XNT_tmp, N_PANDA_tmp = result
            
            # Store LZ 
            N_LZ[i, j] = N_LZ_tmp
            Tstat_LZ[i, j] = t_mu_a(0.12, N_LZ_tmp, N_LZ_sm) # 12% 8B uncertainty
            
            # Store XNT
            N_XNT[i, j] = N_XNT_tmp
            Tstat_XNT[i, j] = t_mu_a(0.12, N_XNT_tmp, N_XNT_sm)
            
            # Store PandaX
            N_PANDA[i, j] = N_PANDA_tmp
            Tstat_PANDA[i, j] = t_mu_a(0.12, N_PANDA_tmp, N_PANDA_sm)

    print("\nScan complete. Saving results...")


    # Pack all data into a pandas DataFrame
    df = pd.DataFrame({
        'eps': epsGrid.flatten(),
        'eta': etaGrid.flatten(),
        'N_LZ': N_LZ.flatten(),
        'Tstat_LZ': Tstat_LZ.flatten(),
        'N_XNT': N_XNT.flatten(),
        'Tstat_XNT': Tstat_XNT.flatten(),
        'N_PANDA': N_PANDA.flatten(),
        'Tstat_PANDA': Tstat_PANDA.flatten()
    })


    # Save to CSV
    job_names = {1: 'ee', 2: 'mumu', 3: 'tautau', 4: 'emu', 5: 'etau', 6: 'mutau'}
    job = job_names.get(nsi_elem, 'unknown')
    csv_file = os.path.join(outpath, f"NR_scan_cluster_{job}.csv")
    df.to_csv(csv_file, index=False)
    print(f"Data successfully saved into file: {csv_file}")


# Run main
if __name__ == "__main__":
    main()