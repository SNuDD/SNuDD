"""Provides electron binding functionality in ElectronBinder."""
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from snudd.config import get_data

BINDING_XENON = pd.read_csv(get_data("binding_energies/binding_xenon.csv")).to_numpy().T
BINDING_ARGON = pd.read_csv(get_data("binding_energies/binding_argon.csv")).to_numpy().T


@dataclass
class ElectronBinder:
    """Dataclass for collecting electron binding energy data and providing a step-function electron counter."""

    orbitals: List[str]
    binding_energies: List[float]  # Binding energies in eV
    orbital_electrons: List[int]

    def available_electrons(self, E_R):
        """Return step function representing effective number of electrons available for scattering."""
        binding = np.tile(self.binding_energies, (len(E_R), 1))
        electron_index = np.less(binding, np.array([E_R * 1e9]).T)  # Electron available if binding > E_R (E_R in GeV!)
        available_electrons = (self.orbital_electrons * electron_index).sum(axis=1)  # Sum electrons free to scatter
        return available_electrons.astype(float)  # Cast as a float instead of inheriting dtype=object


binding_xe = ElectronBinder(*BINDING_XENON)
binding_ar = ElectronBinder(*BINDING_ARGON)

