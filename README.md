# SNuDD

[![arXiv](http://img.shields.io/badge/arXiv-2302.12846-B31B1B.svg)](https://arxiv.org/abs/2302.12846)

**SNuDD** (**S**olar **N**e**u**trinos for **D**irect **D**etection) is a python package for accurate computations of solar neutrino scattering rates at direct detection (DD) experiments in the presence of non-standard neutrino interactions (NSI). 
**SNuDD**  was developed and utilised for the NSI sensitivity estimates of the xenon-based DD experiments XENON, LUX-ZEPLIN and DARWIN in [*A direct detection view of the neutrino NSI landscape*](https://arxiv.org/abs/2302.12846).

When using **SNuDD**, please cite:

D. W. P. Amaral, D. Cerdeno, A. Cheek and P. Foldenauer, \
*A direct detection view of the neutrino NSI landscape*,\
[arXiv:2302.12846 [hep-ph]](https://arxiv.org/abs/2302.12846).



## Prerequisites

**SNuDD** does not have any external dependencies. It relies, however, on the python modules 

- `numpy`: v>= 1.20.3
- `matplotlib`: v>=3.3.4
- `scipy`: v>=1.6.2
- `pandas`: v>=1.3.0
- `setuptools`: v>=52.0.0
- `pymultinest`: v>=2.11



## Installation

You can obtain the sources directly from the [github repository](https://github.com/SNuDD/SNuDD) either by using `git`:
```bash
git clone https://github.com/SNuDD/SNuDD.git
```
or by downloading them as a tarball:
```bash
wget https://github.com/SNuDD/SNuDD/archive/master.tar.gz
tar -xzvf master.tar.gz
mv SNuDD-main SNuDD
```

**SNuDD** can be locally installed from within the directory containing the `SNuDD` repository by calling:
```bash
pip install -e SNuDD
```

## Usage
Once installed, **SNuDD** can be included in your Python code via
```python
import snudd
```

In lieu of a manual, we have created a set of example jupyter notebooks in the `notebooks` sub-directory that explain the basic functionality of **SNuDD** and we recommend going through them.

### [probability_density.ipynb](https://github.com/SNuDD/SNuDD/blob/main/notebooks/probability_density.ipynb):

This notebook goes through the basic steps of how to generate an `GeneralNSI` model object and compute the solar neutrino survival and disappearance probabilites via the `ProbabilityCalculator`. Next, the construction of the solar neutrino denisty matrix via the `DensityMatrixCalculator` from a `GeneralNSI` object is explained. Finally, the difference betwen the denisty matrix elements in the SM and an example NSI are illustrated.

### [CEVNS_rate.ipynb](https://github.com/SNuDD/SNuDD/blob/main/notebooks/CEVNS_rate.ipynb):

This notebook illustrates how to compute the theoretical CEvNS rate (i.e. without taking into account detector effects) in a direct detection experiment. First, a `Nucleus` object is created, which contains all the functions for calculating the CEvNS spectra. This `Nucleus` object is then passed a SM `GeneralNSI` instance to compute the solar neutrino CEvNS rate in the SM in Xenon by calling the `Nucleus` method `spectrum`. In a second step, the `Nucleus` object is updated with a `GeneralNSI` object containing non-zero NSI elements to illustrated the difference in the expected CEvNS rate.

### [EVES_rate.ipynb](https://github.com/SNuDD/SNuDD/blob/main/notebooks/EVES_rate.ipynb):

Analogously, this notebook illustrates how to compute the theoretical EvES rate (again without taking into account detector effects) in a direct detection experiment. After setting up the Xenon `Nucleus` object an `Electron` object is created, which contains all the functions for calculating the EvES spectra. This `Electron` object is then passed a SM `GeneralNSI` instance to compute the solar neutrino EvES rate in the SM in Xenon by calling the `Electron` method `spectrum`. In a second step, the `Electron` object is updated with a `GeneralNSI` object containing non-zero NSI elements to illustrated the difference in the expected EvES rate.

### [scan_Xnt_LZ_2022.ipynb](https://github.com/SNuDD/SNuDD/blob/main/notebooks/scan_Xnt_LZ_2022.ipynb):

Finally, this script illustrates how to perform a grid scan in the NSI paramter space using **SNuDD**. Therefore, a 2D sample grid of NSI values $\varepsilon$ and charged-plane angles $\varphi$ is created. Next, looping over this grid the corresponding number of events for the **LZ 2022** and **XENONnT 2022** exposures is computed, including detecto effects. Lastly, from the 2D grid of events a rough NSI exclusion plot is generated.

## Reporting bugs

**SNuDD** is a work in progress and so far is in an *alpha* release stage. If you find any bugs, please report them by creating an `Issue` on the project [GitHub](https://github.com/SNuDD/SNuDD) page.
