"""Creates efficiency functions for DD experiments, which can be extended to a new threshold"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import root_scalar, brentq

from snudd import config

LZ_NR_DATA = np.loadtxt(config.get_data("exps/lz/LZ_NR.dat"), unpack=True)
LZ_ER_DATA = np.loadtxt(config.get_data("exps/lz/LZ_ER.dat"), unpack=True)
XNT_NR_DATA = np.loadtxt(config.get_data("exps/xnt/XNT_NR.dat"), unpack=True)
XNT_ER_DATA = np.loadtxt(config.get_data("exps/xnt/XNT_ER.dat"), unpack=True)



LZ_NR_22 =np.loadtxt(config.get_data("exps/lz/LZ_NR_first_data.txt"), unpack=True)
LZ_NR_22 = np.array([LZ_NR_22[0][LZ_NR_22[1] > 0.0], LZ_NR_22[1][LZ_NR_22[1] > 0.0]]) ### get rid of the zero entries because they mess up with the interpolation
XNT_ER_22 = np.loadtxt(config.get_data("exps/xnt/XNT_ER_22.csv"), delimiter=',', unpack=True)

##### below is the LZ efficiency from 
LZ_B8_DATA = np.loadtxt(config.get_data("exps/LZ8B2025/LZB8-v2-Efficiency.csv"), delimiter=",", skiprows=7, unpack=True)


def linear_extend_strategy(E_R, E_thresh_50_new, efficiency):
    """Return linearly extended efficiency."""
    shift = efficiency.threshold_50 - E_thresh_50_new

    condlist = [efficiency.E_R_min - shift <= E_R <= efficiency.threshold_plateau_E_R - shift,
                efficiency.threshold_plateau_E_R - shift < E_R <= efficiency.threshold_plateau_E_R,
                efficiency.threshold_plateau_E_R < E_R <= efficiency.E_R_max,
                ]

    funclist = [lambda E: efficiency(E + shift),
                lambda E: efficiency(efficiency.threshold_plateau_E_R),
                lambda E: efficiency(E),
                0.]

    return np.piecewise(E_R, condlist=condlist, funclist=funclist)


def log_extend_strategy(E_R, E_thresh_50_new, efficiency):
    """Return log extended efficiency function."""
    shift = E_thresh_50_new / efficiency.threshold_50

    condlist = [(efficiency.E_R_min * shift <= E_R) & (E_R <= efficiency.threshold_plateau_E_R * shift),
                (efficiency.threshold_plateau_E_R * shift < E_R) & (E_R <= efficiency.threshold_plateau_E_R),
                (efficiency.threshold_plateau_E_R < E_R) & (E_R <= efficiency.E_R_max),
                ]

    funclist = [lambda E: efficiency(E / shift),
                lambda E: efficiency(efficiency.threshold_plateau_E_R),
                lambda E: efficiency(E),
                0.]

    return np.piecewise(E_R, condlist=condlist, funclist=funclist)


class Efficiency:
    """An interpolant that returns the efficiency function along with some of its important properties. An extension
    of the efficiency is also available.
    """

    def __init__(self, data, order=1):
        self.data = data
        self.order = order
        self._interpolant = InterpolatedUnivariateSpline(data[0] / 1e6, data[1], k=self.order)

    @property
    def E_R_min(self):
        """Set minimum recoil energy to be minimum provided energy in data."""
        return self.data[0][0] / 1e6  # Division to get into GeV

    @property
    def E_R_max(self):
        """Set maximum recoil energy to be maximum provided energy in data."""
        return self.data[0][-1] / 1e6  # Division to get into GeV

    @property
    def threshold_50(self) -> float:
        """Solves for 50% efficiency threshold"""
        return brentq(lambda E_R: self._interpolant(E_R) - 0.5, a=self.E_R_min, b=self.threshold_plateau_E_R)

    @property
    def threshold_plateau_E_R(self):
        """Return energy at which efficiency curve begins to plateau."""
        efficiency_derivative = self._interpolant.derivative()
        E_R_middle = 10 ** np.mean((np.log10(self.E_R_max), np.log10(self.E_R_min)))  # The log-middle E_R
        return root_scalar(efficiency_derivative, x0=E_R_middle * 0.2, x1=E_R_middle * 2, xtol=9e-7).root

    def extended_efficiency(self, E_R, E_thresh_50_new, extend_strategy=log_extend_strategy):
        """Return efficiency function extended to new, lower 50% threshold energy as per some strategy"""
        return extend_strategy(E_R, E_thresh_50_new, efficiency=self)

    def __call__(self, E_R):
        """Return interpolant if within interpolation range, else return 0."""

        return np.piecewise(E_R,
                            condlist=[(E_R <= self.E_R_max) & (E_R >= self.E_R_min), ],
                            funclist=[lambda E: self._interpolant(E), 0.])


efficiency_lz_nr = Efficiency(LZ_NR_DATA, order=2)
threshold_50_lux_nr = efficiency_lz_nr.threshold_50  # TODO: Replace these with field call

efficiency_lz_er = Efficiency(LZ_ER_DATA)
threshold_50_lux_er = efficiency_lz_er.threshold_50

efficiency_xnt_nr = Efficiency(XNT_NR_DATA)
threshold_50_xnt_nr = efficiency_xnt_nr.threshold_50

efficiency_xnt_er = Efficiency(XNT_ER_DATA)
threshold_50_xnt_er = efficiency_xnt_er.threshold_50

efficiency_lz_b8 = Efficiency(LZ_B8_DATA)
#threshold_50_lz_b8 = efficiency_lz_b8.threshold_50



######################
efficiency_lz_nr_22 = Efficiency(LZ_NR_22)
threshold_50_lux_nr_22 = efficiency_lz_nr_22.threshold_50 


efficiency_xnt_er_22 = Efficiency(XNT_ER_22)
threshold_50_xnt_er_22 = efficiency_xnt_er_22.threshold_50

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    E_Rs_data_nr, eff_lz_data_nr = np.loadtxt(config.get_data("exps/lz/LZ_NR.dat")).T
    E_Rs_nr = np.logspace(np.log10(E_Rs_data_nr[0]), np.log10(E_Rs_data_nr[-1]), 1000) / 1e6

    E_Rs_data_er, eff_lz_data_er = np.loadtxt(config.get_data("exps/lz/LZ_ER.dat")).T
    E_Rs_er = np.logspace(np.log10(E_Rs_data_er[0]), np.log10(E_Rs_data_er[-1]), 1000) / 1e6

    E_Rs = np.logspace(-2, 2, 1000) / 1e6
    plt.figure()
    plt.loglog(E_Rs_er * 1e6, efficiency_xnt_er(E_Rs_er))
    plt.loglog(E_Rs * 1e6, efficiency_xnt_er.extended_efficiency(E_Rs, 1e-7), ls='--')
    plt.show()
    plt.figure()
    plt.semilogx(E_Rs_er, efficiency_xnt_er._interpolant.derivative()(E_Rs_er))
    print(threshold_50_lux_nr)
    print(threshold_50_lux_er)

    print(threshold_50_xnt_nr)
    print(threshold_50_xnt_er)

    plt.show()
