"""Supplies convolution functionality through resolution functions."""
import warnings

import numpy as np
from nuddnsi.efficiencies import Efficiency, efficiency_lz_nr, efficiency_lz_er, efficiency_xnt_nr, efficiency_xnt_er
from nuddnsi.quenching import Quenching, quenching_xe, quenching_electron
from scipy.special import erf


def resolution_lux(E_R):
    """Energy resolution (fractionally) as a function of the energy for LUX-ZEPLIN"""
    a = 0.33 * 1e-3  # Last factor is to convert from keV^{1/2} to GeV^{1/2}
    return a / np.sqrt(E_R)


def resolution_xnt(E_R):
    """Energy resolution (fractionally) as a function of the energy for Xenon N Tonne"""
    a = 0.310 * 1e-3  # Last factor is to convert from keV^{1/2} to GeV^{1/2}
    b = 0.0037
    return a / np.sqrt(E_R) + b


class Resolution:
    """Applies resolution relations"""

    def __init__(self, resolution_fn, efficiency, quenching):
        """To get resolution curve (in kev_ee!!)"""
        self.resolution_fn = resolution_fn
        self.efficiency = efficiency
        self.quenching = quenching  # quenching needed if efficiency function in kev_nr

    def extended_resolution(self, E_R, E_thresh_50_new):
        """Extend resolution function below original 50% efficiency point by capping it at that value."""
        shift = self.quenching.convert_nr2ee(self.efficiency.threshold_50) - E_thresh_50_new

        condlist = [E_R < E_thresh_50_new, ]
        funclist = [lambda E: self(self.quenching.convert_nr2ee(self.efficiency.threshold_50)),
                    lambda E: self.resolution_fn(E + shift)]

        return np.piecewise(E_R, condlist, funclist)

    def __call__(self, E_R):
        """Calculate sigma as it is, with no shifts or extensions"""
        return self.resolution_fn(E_R)


class Convolver:
    """Supplies convolution functionality given a total spectrum (total signal + background) and extended energy values.
    Can give the convolved spectrum or can skip convolution integral entirely to calculate binned rate.
    """

    def __init__(self, E_primes, spectrum, efficiency: Efficiency, resolution: Resolution, quenching: Quenching = None):
        self.E_primes = E_primes
        self.spectrum = spectrum
        self.efficiency = efficiency
        self.resolution = resolution
        self.quenching = quenching if quenching is not None else quenching_electron
        self._E_primes_ee = self.quenching.convert_nr2ee(E_primes)

    def convolved_binned_rate(self, E_1, E_2):
        """Return the convolved rate within a bin with edges E_1 < E_2."""
        return np.trapz(self._energy_response_integrand(E_1, E_2), self._E_primes_ee * 1e6)

    def convolve_spectrum(self, E_R):
        """Return convolved spectrum at E_R"""
        convolution = np.array([np.trapz(self._convolution_integrand(E), self._E_primes_ee) for E in E_R])
        return spec_ee2nr(E_R, convolution, self.quenching)

    def _energy_response_function(self, E_ee, E_1, E_2):
        """Calculate energy response function needed for quick post-convolution rate calculation (Le Trick)."""
        sigma = self.resolution(E_ee) * E_ee
        E_1_ee = self.quenching.convert_nr2ee(E_1)
        E_2_ee = self.quenching.convert_nr2ee(E_2)
        erf1 = erf((E_1_ee - E_ee) / (np.sqrt(2) * sigma))
        erf2 = erf((E_2_ee - E_ee) / (np.sqrt(2) * sigma))
        return 0.5 * (erf2 - erf1)

    def _energy_response_integrand(self, E_1, E_2):
        """Calculate the integrand of the energy response integral, ready for integration."""
        self._check_energy_endpoints([E_1, E_2])  # Catch potential introduction of inaccuracies

        efficiencies = self.efficiency(self.E_primes)
        energy_responses = self._energy_response_function(self._E_primes_ee, E_1, E_2)
        spectrum_ee = spec_nr2ee(self.E_primes, self.spectrum, self.quenching)

        return spectrum_ee * efficiencies * energy_responses

    def _response_function(self, E_prime_ee, E_ee, Z=5):
        """The response function to convolve with. Only Gaussian implemented."""
        sigma = self.resolution(E_prime_ee) * E_prime_ee
        condlist = [E_ee - Z * sigma >= E_prime_ee]
        funclist = [0., lambda E: gaussian_response(E_ee, E, self.resolution(E) * E)]
        return np.piecewise(E_prime_ee, condlist, funclist)

    def _convolution_integrand(self, E_R):
        """Calculate the integrand of the convolution integral, ready for integration."""
        self._check_energy_endpoints(E_R)  # Catch potential introduction of inaccuracies
        efficiencies = self.efficiency(self.E_primes)
        E_ee = self.quenching.convert_nr2ee(E_R)
        responses = self._response_function(self._E_primes_ee, E_ee)
        spectrum_ee = spec_nr2ee(self.E_primes, self.spectrum, self.quenching)
        return spectrum_ee * efficiencies * responses

    def _check_energy_endpoints(self, E_R):
        """Raise a warning if desired recoil energy range is beyond that of the supplied spectrum, introducing
        inaccuracies to the convolution.
        """
        if np.min(E_R) < np.min(self.E_primes):
            warnings.warn(f"Minimum of supplied recoil energy ({np.min(E_R)}) is less than "
                          f"minimum of supplied E_prime ({np.min(self.E_primes)}). Convolution may not be accurate.")

        elif np.max(E_R) > np.max(self.E_primes):
            warnings.warn(f"Maximum of supplied recoil energy ({np.max(E_R)}) is greater than "
                          f"maximum of supplied E_prime ({np.max(self.E_primes)}). Convolution may not be accurate.")


def gaussian_response(E, mu, sigma):
    """Return gaussian response function centered at mu with width sigma"""
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(E - mu) ** 2 / (2 * sigma ** 2))


def spec_nr2ee(E_nr, spec_nr, quenching):
    """To convert from keV_nr to keV_ee spec for NRs. Specifically for xenon."""
    conv_factor = 1 / (quenching(E_nr) + quenching.derivative(E_nr) * E_nr * 1e6)
    return spec_nr * conv_factor


def spec_ee2nr(E_nr, spec_ee, quenching):
    """To convert from ee to nr spec for NRs"""
    conv_factor = (quenching(E_nr) + quenching.derivative(E_nr) * E_nr * 1e6)
    return spec_ee * conv_factor


res_lz_nr = Resolution(resolution_lux, efficiency_lz_nr,quenching_xe)
res_lz_er = Resolution(resolution_lux, efficiency_lz_er, quenching_electron)

res_xnt_nr = Resolution(resolution_xnt, efficiency_xnt_nr,quenching_xe)
res_xnt_er = Resolution(resolution_xnt, efficiency_xnt_er, quenching_electron)

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    E_Rs = np.linspace(0.001, 10, 1000) / 1e6
    res = Resolution(resolution_xnt, efficiency_xnt_nr)
    res_fn = res.resolution_fn(E_Rs)
    resolutions_vanilla = res(E_Rs)
    resolutions_capped = res.extended_resolution(E_Rs, efficiency_xnt_nr.threshold_50)
    resolutions_extended = res.extended_resolution(E_Rs, 2 / 1e6)
    plt.figure()
    plt.plot(E_Rs * 1e6, resolutions_vanilla, c='b', label="Uncapped", ls=':')
    plt.plot(E_Rs * 1e6, resolutions_capped, ls='--', label=r"Capped and $E_{\mathrm{th}} \simeq 4\,\mathrm{keV}$",
             c='r')
    plt.plot(E_Rs * 1e6, resolutions_extended, c='g', label=r"Capped and $E_{\mathrm{th}} = 2\,\mathrm{keV}$",
             ls='-.')
    plt.ylim(0, 1)
    plt.xlim(0, 10)
    plt.ylabel(r"$\sigma / E_R$")
    plt.xlabel(r"$E_R\,\mathrm{(keV)}$")
    plt.legend(frameon=False)
    plt.show()
