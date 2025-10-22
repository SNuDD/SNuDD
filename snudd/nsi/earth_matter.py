"""Earth Matter effects for Neutrino Oscillations with NSI."""



from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Callable, Sequence, Optional
from snudd.nsi import flux_dists, oscillation as osc
import numpy as np

# ---------- Constants (natural units) ----------
CF = 5.06773e18         # km -> GeV^-1
s2GFNa = 7.6326e-23      # Sqrt[2]  UC[GF *NA *cm^-3, GeV ]
RE_KM, ATM_KM = 6371.0, 15.0


# ---------- Interfaces ----------
class EarthModel(Protocol):
    def rhoYe_gcm3(self, r_over_RE: float) -> float:
        """Return Ye * rho(r) in g/cm^3 at r/RE ∈ [0,1]."""


# 1) Direct callable hook ------------------------------------------------------
@dataclass
class CallableEarth(EarthModel):
    f: Callable[[float], float]  # f(r_over_RE) -> Ye*rho [g/cm^3]
    def rhoYe_gcm3(self, r_over_RE: float) -> float:
        r = float(np.clip(r_over_RE, 0.0, 1.0))
        return float(self.f(r))


# 2) Layered polynomial (PREM-like) -------------------------------------------
@dataclass
class LayeredPolyEarth(EarthModel):
    """
    User supplies:
      - xr_km: layer boundaries (km), monotonically increasing (len L+1)
      - coeffs: list of (a,b,c,d) for each layer (len L), evaluated in r = (radius / RE)
      - Ye_core, Ye_mantle: piecewise-constant Ye (or pass a custom Ye(r/RE) via ye_fn)
    """
    xr_km: Sequence[float]
    coeffs: Sequence[tuple]  # [(a,b,c,d), ...] one per layer
    Ye_core: float = 0.466
    Ye_mantle: float = 0.494
    ye_fn: Optional[Callable[[float], float]] = None  # overrides Ye_core/mantle if given

    def __post_init__(self):
        xr = np.asarray(self.xr_km, dtype=float)
        if not (np.all(np.isfinite(xr)) and np.all(np.diff(xr) > 0)):
            raise ValueError("xr_km must be strictly increasing and finite")
        if len(self.coeffs) != len(xr) - 1:
            raise ValueError("coeffs length must be len(xr_km)-1")
        self.xr = xr
        self.RE_KM = 6371.0

    def _layer_index(self, R_km: float) -> int:
        # right-edge binning; clamp to last layer
        k = int(np.searchsorted(self.xr, R_km, side="right") - 1)
        return max(0, min(k, len(self.xr) - 2))

    def _Ye(self, r: float) -> float:
        if self.ye_fn is not None:
            return float(self.ye_fn(r))
        # default: core vs mantle split at r = 0.546 like classic PREM usage
        return self.Ye_core if r <= 0.546 else self.Ye_mantle

    def rhoYe_gcm3(self, r_over_RE: float) -> float:
        r = float(np.clip(r_over_RE, 0.0, 1.0))
        R_km = r * self.RE_KM
        k = self._layer_index(R_km)
        a, b, c, d = self.coeffs[k]
        rho = a + b*r + c*r*r + d*r*r*r  # g/cm^3 (mass density)
        return self._Ye(r) * rho

# 3) Tabulated profile with spline --------------------------------------------
class TabulatedEarth(EarthModel):
    """
    Provide tabulated radius and Ye*rho values:
      - r_nodes: array of r/RE in [0,1]
      - rhoYe_nodes: array of Ye*rho (g/cm^3)
    Uses a monotone cubic spline (PCHIP-like) via numpy interp fallback + local slopes.
    """
    def __init__(self, r_nodes: Sequence[float], rhoYe_nodes: Sequence[float]):
        r = np.asarray(r_nodes, dtype=float)
        y = np.asarray(rhoYe_nodes, dtype=float)
        if r.ndim != 1 or y.ndim != 1 or len(r) != len(y) or len(r) < 2:
            raise ValueError("r_nodes and rhoYe_nodes must be 1D, same length >=2")
        if np.any((r < 0) | (r > 1)) or np.any(~np.isfinite(y)) or np.any(~np.isfinite(r)):
            raise ValueError("r_nodes must be in [0,1] and finite; rhoYe_nodes finite")
        order = np.argsort(r)
        self.r = r[order]
        self.y = y[order]

    def rhoYe_gcm3(self, r_over_RE: float) -> float:
        r = float(np.clip(r_over_RE, 0.0, 1.0))
        # simple, safe interpolation; replace with scipy PchipInterpolator if you like
        return float(np.interp(r, self.r, self.y))



class EarthPropagator:
    def __init__(self,  model, osc_params=osc.osc_params_best, 
                 earthmodel: Optional[EarthModel] = None, Nst: int = 200, Nav: int = 200):
        self.osc_params = osc_params
        self.model= model
        self.earthmodel = earthmodel  # can be any EarthModel
        self.Nst = int(Nst); self.Nav = int(Nav)
        self.ERad = CF * RE_KM
        self.ARad = CF * ATM_KM
        self.tRad = self.ERad + self.ARad




    def _vacuum_H(self, Enu: float) -> np.ndarray:
        U = osc.UPMNS(self.osc_params)
        diag = np.diag([0.0, self.osc_params.delta_m12/(2*Enu), self.osc_params.delta_m31/(2*Enu)])
        return U @ diag @ U.conj().T

    def _V_matrix(self) -> np.ndarray:
        epsmat = self.model.eps_matrix
        return (np.array([[1.0 , 0.0 , 0.0], 
                   [0.0, 0.0 ,0.0], 
                   [0.0, 0.0, 0.0 ]], dtype=np.complex128) + epsmat)

    def _r_over_RE_along_chord(self, x: float, ceta: float) -> float:
        norm = 1.0 / (RE_KM + ATM_KM)
        root_common = np.sqrt(max(0.0, 1.0 - (norm*RE_KM)**2 * (1 - ceta*ceta)))
        r_dimless = np.sqrt(max(0.0, 1 + x*x - 2*x*root_common)) / (norm*RE_KM)
        return r_dimless

    def S_matrix(self, Enu_GeV: float, ceta: float) -> np.ndarray:
        if np.arccos(ceta) >= np.pi/2:
            return np.eye(3, dtype=np.complex128)

        Enu = Enu_GeV 
        Hv = self._vacuum_H(Enu)
        Vf = self._V_matrix()

        t1 = (self.ERad*ceta + np.sqrt(max(0.0, self.tRad*self.tRad
                  - self.ERad*self.ERad*(1 - ceta*ceta)))) / self.tRad

        S = np.eye(3, dtype=np.complex128)
        for k in range(self.Nst):
            xmin = (t1 * k) / self.Nst
            xmax = (t1 * (k+1)) / self.Nst
            # average Ye*rho for this slice from whichever model was provided
            Vav = 0.0
            for l in range(self.Nav + 1):
                xi = xmin + (xmax - xmin) * (l/(self.Nav + 1))
                r_over_RE = self._r_over_RE_along_chord(xi, ceta)
                Vav += self.earthmodel.rhoYe_gcm3(r_over_RE)
            Vav /= (self.Nav + 1)

            H = Hv + s2GFNa * Vav * Vf
            Em, Um = np.linalg.eigh(H)
            phase = np.exp(-1j * Em * self.tRad * (xmax - xmin))
            S = S @ (Um @ np.diag(phase) @ Um.conj().T)
        return S