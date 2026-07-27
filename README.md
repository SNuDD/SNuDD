
<p align="center">
    <img src="https://github.com/SNuDD/SNuDD/blob/main/snudd_logo.png" alt="Description" width="400">
</p>

[![arXiv](http://img.shields.io/badge/arXiv-2302.12846-B31B1B.svg)](https://arxiv.org/abs/2302.12846)

We present **SNuDD** (**S**olar **N**e**u**trinos for **D**irect **D**etection): a Python-based, open-source codebase for accurately computing solar neutrino scattering rates due to neutrino-electron and -nucleus scattering. **SNuDD** can be used to compute the SM scattering rates and the modified rates in the presence of BSM physics, such as in the form of non-standard interactions (NSI). **SNuDD** has been developed to consistently incorporate BSM neutrino physics effects both in neutrino propagation (solar and terrestrial) and in neutrino scattering within the detector. Additionally, **SNuDD** provides functionality to incorporate detector effects like energy thresholds, selection efficiencies, and resolution effects for generating realistic signal spectra.

Below, we describe the main functionality of **SNuDD**: from specifying an NSI model, to computing the neutrino density matrix, to calculating the scattering rate with experimental effects included. The code is freely available
under an open-source license. All code examples shown here are bundled in a Jupyter notebook named `quick_start.ipynb` and can be run interactively.


When using **SNuDD**, please cite:

- D. W. P. Amaral, D. Cerdeno, A. Cheek and P. Foldenauer, \
*A direct detection view of the neutrino NSI landscape*,\
[arXiv:2302.12846 [hep-ph]](https://arxiv.org/abs/2302.12846).



## Prerequisites

**SNuDD** does not have any external dependencies. It relies, however, on the python modules 

- `numpy`: v>= 1.22
- `scipy`: v>=1.8
- `pandas`: v>=1.3
- `numba`: v>0.57    (optional for minor speed-up)

In order to run the `quick_start.ipynb` notebook, the following packages are needed

- `ipykernel`
- `matplotlib`



## Installation

You can obtain the sources directly from the [github repository](https://github.com/SNuDD/SNuDD) by using `git`:
```bash
git clone https://github.com/SNuDD/SNuDD.git
```


**SNuDD** can be locally installed from within the `SNuDD` repository by calling:
```bash
pip install (-e) .
```
Use the `-e` option for an editable installation.

## Usage
Once installed, **SNuDD** can be included in your Python code via
```python
import snudd
```

We have created an example notebook in the `notebooks` sub-directory named `quick_start.ipynb` that explain the basic functionality of **SNuDD** and we recommend going through them.

### Setting Up a Model:

First, we set up an NSI model. To specify a model, we must provide the $3\times3$ matrix of the NSI magnitudes in flavour space $\varepsilon^{\eta,\varphi}_{\alpha\beta}$, as well as the relative NSI strength with electrons, protons, and neutrons, which is encoded in the angles $\eta$ and $\varphi$. For the definition of the parametrisation used in the code, please refer to [arXiv:2302.12846 [hep-ph]](https://arxiv.org/abs/2302.12846).

In the following example code, we show how to set up an NSI model instance for a purely off-diagonal $\mu\tau$-coupling of magnitude $\varepsilon^{\eta,\varphi}_{\mu\tau}=0.1$, equal coupling strengths with protons and neutrons ($\eta=\pi/4$), and no coupling with electrons ($\varphi=0$).

```python
from snudd.models import GeneralNSI

# Define NSI parameters
NSI_matrix = numpy.array([[0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.1],
                          [0.0, 0.1, 0.0]])
NSI_eta = numpy.pi/4
NSI_phi = 0

# Create NSI model object
NSI_model = GeneralNSI(NSI_matrix, NSI_eta, NSI_phi)
```

### Calculating the Density Matrix:

Next, we demonstrate how to compute the solar neutrino density matrix, both for day-time (only solar matter evolution) and night-time observation (combined solar + Earth-matter evolution). We note that the base unit of energy in **SNuDD** is GeV. Hence, care should be taken when specifying neutrino energies and recoil energies.


The following lines of code demonstrate how to setup a `DensityMatrixCalculator` object and calculate the density matrix elements over a range of neutrino energies. This simple call of the `density` method will only compute the day-time density matrix, in this case for the 8B neutrinos.

```python
from snudd.nsi.nsi_probabilities import DensityMatrixCalculator

# Create probability calculator 
NSI_density_calculator = DensityMatrixCalculator(NSI_model)

# Calculate solar density matrix over neutrino energy range
Enus = numpy.geomspace(1e-6, 1e0, 200) # in GeV
solar_density = NSI_density_calculator.density(Enus, '8B')
```

#### Including Earth-matter Effects:

We can include earth-matter effects in the solar neutrino propagation. For this, we need to specify a detector location and track the solar neutrino incident angles (or equivalently their paths through Earth's interior) over the experimental data taking period. **SNuDD** provides the `SolarAngles` module, allowing the user to compute a weighted histogram of neutrino incident angles (outputting $\cos\ \eta_{\rm nad}$) over a given data taking period at a specified detector location.

In the following example, we show how to generate the histogram of incident angles for the Gran Sasso location (XENONnT) for a data taking campaign of one year:

```python
from snudd.geometry import SolarAngles

# Define experimental parameters (location, data taking period)
lat_gs = 42.47  # Latitude of Gran Sasso in degrees north
t0     = 0      # t0 - beginning in days after 3rd January (perihelion)
T      = 365    # T  - duration of data taking in days

# Define solar angles calculator for Gran Sasso
GranSasso = SolarAngles(latitude=lat_gs, t0=t0, T=T)

# Generate histogram of cos(nadir) 
cnadirs, weights = GranSasso.cnadir_hist(bins=30)
```

With this weighted histogram of neutrino incident angles, we can compute the full neutrino density matrix by averaging the Earth-matter effects over the incident angles using the `DensityMatrixCalculator` as follows:


```python
# Calculate full density matrix over neutrino energy range
Enus = numpy.geomspace(1e-6, 1e0, 200) # in GeV
earth_density = NSI_density_calculator.density_earth(Enus, cnadirs, weights, nu='8B')
```


### Computing the Neutrino Recoil Spectrum:


In **SNuDD**, the full pipeline for computing 
the neutrino recoil spectrum via the trace formalism
is implemented in the `Target` class. More precisely, its two subclasses
`Nucleus`
and
`Electron`
enable the user to define scattering target objects for calculating nuclear and electron recoil event rates, respectively. 



#### CEvNS spectrum:
The following code block illustrates how to generate a spectrum for coherent elastic neutrino-nucleus scattering (CEvNS).
To do so, we first have to specify the nuclear scattering target via a `Nucleus` object. We then have to update it with our current `GeneralNSI` model, which holds the expressions for the relevant CEvNS cross section. Next, we pre-compute the neutrino density matrix via the call `prepare_density` including the Earth-matter evolution. Note that simply calling `prepare_density()` without arguments computes only the day-time density without any Earth-matter effects.
Finally, we compute the CEvNS spectrum over a predefined range of recoil energies (specified in GeV).


```python
from snudd import config
from snudd.targets import Nucleus

# Create scattering target
Xe_nucleus = Nucleus(54, 132, mass=131.9041535 * config.u) # single isotope 

# Load NSI model in scattering target and generate density matrix
Xe_nucleus.update_model(NSI_model)
Xe_nucleus.prepare_density(cnadirs=cnadirs, cnadir_weights=weights)

# Compute recoil spectrum
E_Rs = numpy.geomspace(1e-2, 1e2, 500) / 1e6  # Recoil energy in GeV
NSI_spec_nr = Xe_nucleus.spectrum(E_Rs)
```

#### EvES spectrum:

Computing accurate spectra for elastic neutrino-electron scattering (EvES) requires knowledge of the available electrons for scattering at a given neutrino energy.  **SNuDD**'s `Electron` object requires information about the host nucleus and its respective orbital binding energies. For a generic atom, the user can calculate the resulting electron recoil spectra according to the stepping approximation. For xenon targets, one can instead produce a spectrum scaled according to the relativistic random phase approximation (RRPA). The handling of the binding energies in the appropriate scattering prescription is implemented in the `Electron` class, and we illustrate how to generate an electron recoil spectrum in the following block of code.

As for CEvNS, we first setup a `Nucleus` object for the host atom, which we use together with a `binding` object holding the orbital binding energies and an `rrpa_scaling` object to initialize an `Electron` object. As before, we a to feed the `Electron` object to current `GeneralNSI` model and pre-compute the neutrino density matrix. Finally, we can generate the EvES spectrum over a range of recoil energies.


```python
from snudd import config
from snudd.targets import Nucleus, Electron   # still requires the host nucleus
from snudd.binding import binding_xe          # dataclass of binding energy data of xenon 
from snudd.rrpa    import rrpa_scaling        # rrpa scaling for bound electron

# Create host nucleus
Xe_nucleus = Nucleus(54, 132, mass=131.9041535 * config.u) # single isotope 

# Create bound electron object
Xe_electron = Electron(Xe_nucleus, binding_xe, rrpa_scaling) 
Xe_electron.update_model(NSI_model)
Xe_electron.prepare_density()

# Compute recoil spectrum
E_Rs = numpy.geomspace(1e-2, 1e2, 500) / 1e6  # Recoil energy in GeV
NSI_spec_er = Xe_electron.spectrum(E_Rs)
```


### Including Detector Effects:

To make realistic predictions of neutrino scattering rates at direct detection experiments, we apply
an energy-dependent efficiency $\epsilon(E)$ to the theoretical recoil spectrum and convolve it with a detector resolution response function $\Phi$.
**SNuDD** provides the framework for this bundled into the `Convolver` class, and contains pre-defined efficiency and resolution functions for LZ, XENONnT, and PandaX-4T. 

In the following example code, we illustrate how to compute the experimental count rate for solar neutrino-electron scattering in a future xenon experiment modelled after XLZD or PandaX-xT.  We first set up a `Convolver` object and initialize it with the recoil `spectrum` 
evaluated over a range of energies (`E_Rs`), an `efficiency` function, and a `resolution` function. To obtain realistic predictions, we employ the latest LZ efficiency function. By calling the method `convolved_binned_rate(E1, E2)`, we compute the convolved number of signal counts in an energy bin with bin edges $[E_1, E_2]$. Applying this prescription over a predefined set of energy bins, we can generate the experimental binned histogram of EvES events. 


```python
from snudd.efficiencies import efficiency_lz_er_WS24
from snudd.resolution   import res_lz_er
from snudd.resolution   import Convolver

# Counts @ future xenon detector
def counts_future_xe(specturm, E_Rs, E1, E2):
    exposure_future = 200. # t yrs
    convolution_lz_sig = Convolver(E_Rs, specturm, efficiency_lz_er_WS24, res_lz_er)
    return convolution_lz_sig.convolved_binned_rate(E1, E2) * exposure_future

# Definine histogram binning 
bin_edges   = numpy.linspace(1, 20, 20, endpoint=True) / 1e6 # Bin edges in GeV 
bin_width   = bin_edges[1] - bin_edges[0]
bin_centers = bin_edges[:-1] + bin_width/2

# Apply detector effects to neutrino spectrum
counts_er = [counts_future_xe(NSI_spec_er, E_Rs, (bc - bin_width/2), (bc + bin_width/2)) for bc in bin_centers]
```


## Reporting bugs

If you find any bugs, please report them by creating an `Issue` on the project [GitHub](https://github.com/SNuDD/SNuDD) page.
