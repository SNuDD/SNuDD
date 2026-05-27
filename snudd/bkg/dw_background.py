import numpy as np
from scipy import interpolate
from snudd.config import get_data


E_low_ext = 1e-2

### DOUBLE BETA ###
E_R_total, double_bkg = np.loadtxt(get_data("exps/darwin/bkg_double_beta")).T

E_R_total = np.insert(E_R_total, 0, 0.01)
double_bkg = np.insert(double_bkg, 0, 1e-5) / 0.005  # As figure with 99.5% cut

E_R_range = np.logspace(-2, 2, 1000) / 1e6
interp_double = interpolate.interp1d(E_R_total / 1e6, double_bkg, kind='linear')
range_double = interp_double(E_R_range)

### FLAT ###

flat_bkg_rate_scalar = (19 + 565 + 139) / (30 - 2) / 14  # Taken from Table 2 of 1309.7024. Converted to /ton-yr-keVnr.
total_bkg_spec_fn = lambda E_R: flat_bkg_rate_scalar + interp_double(E_R)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    E_Rs = np.logspace(-2, np.log10(50)) / 1e6

    plt.figure()
    plt.loglog(E_Rs * 1e6, total_bkg_spec_fn(E_Rs))
    plt.show()
