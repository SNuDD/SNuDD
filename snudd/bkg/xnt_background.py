import numpy as np
from scipy import interpolate

from snudd.config import trapezoid, get_data


E_R_total, total_bkg = np.loadtxt(get_data("exps/xnt/bkg_xnt.txt")).T
E_R_nu, nu_bkg = np.loadtxt(get_data("exps/xnt/bkg_solar_nu.txt")).T

E_low_ext = 1e-2

E_R_total = np.insert(E_R_total, 0, E_low_ext)
E_R_total = np.insert(E_R_total, len(E_R_total), 100)
total_bkg = np.insert(total_bkg, 0, total_bkg[0])
total_bkg = np.insert(total_bkg, len(total_bkg), total_bkg[-1])

E_R_nu = np.insert(E_R_nu, 0, E_low_ext)
E_R_nu = np.insert(E_R_nu, len(E_R_nu), 100)

nu_bkg = np.insert(nu_bkg, 0, nu_bkg[0])
nu_bkg = np.insert(nu_bkg, len(nu_bkg), nu_bkg[-1])

total_bkg_interp = interpolate.interp1d(E_R_total/1e6, total_bkg)
nu_bkg_interp = interpolate.interp1d(E_R_nu/1e6, nu_bkg)

total_bkg_spec_fn = lambda E_R: total_bkg_interp(E_R) - nu_bkg_interp(E_R)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    E_Rs = np.logspace(-1, np.log10(30)) / 1e6

    plt.figure()
    plt.loglog(E_Rs * 1e6, total_bkg_interp(E_Rs))
    plt.loglog(E_Rs * 1e6, nu_bkg_interp(E_Rs))
    plt.loglog(E_Rs * 1e6, total_bkg_spec_fn(E_Rs))
    plt.show()

    print(trapezoid(total_bkg_spec_fn(E_Rs), E_Rs*1e6))
