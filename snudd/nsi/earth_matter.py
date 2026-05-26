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
    # def rhoYe_gcm3(self, r_over_RE: float) -> float:
    #     """Return Ye * rho(r) in g/cm^3 at r/RE ∈ [0,1]."""
    def rhoYe_gcm3(self, r_over_RE):
        """
        Return Ye * rho(r) in g/cm^3 for a vector of r/RE ∈ [0,1].
        This allows for efficient vectorized evaluation along the neutrino trajectory.
        Must accept scalar OR ndarray input and return same shape output.
        """
        ...

# 1) Direct callable hook ------------------------------------------------------
@dataclass
class CallableEarth(EarthModel):
    f: Callable[[float], float]   # f(r_over_RE) -> Ye*rho [g/cm^3] 

    def rhoYe_gcm3(self, r_over_RE):
        """
        Supports scalar or ndarray inputs.

        Strategy:
        - Convert input to numpy array
        - Apply clipping elementwise
        - Apply user function elementwise
        """
        # Restrict to [0,1]
        r = np.clip(r_over_RE, 0.0, 1.0)

        # If f is not vectorized → apply it elementwise safely
        return np.array([self.f(x) for x in np.ravel(r)]).reshape(np.shape(r))
    


# 2) Layered polynomial (PREM-like) -------------------------------------------
@dataclass
class LayeredPolyEarth(EarthModel):
    """
    User supplies:
      - xr_km: layer boundaries (km), monotonically increasing (len L+1)
      - coeffs: list of (a,b,c,d) for each layer (len L), evaluated in r = (radius / RE)
      - Ye_core, Ye_mantle: piecewise-constant Ye (or pass a custom Ye(r/RE) via ye_fn)
    """
    xr_km:      Sequence[float]
    coeffs:     Sequence[tuple]  # [(a,b,c,d), ...] one per layer
    Ye_core:    float = 0.466
    Ye_mantle:  float = 0.494
    ye_fn:      Optional[Callable[[float], float]] = None  # overrides Ye_core/mantle if given

    def __post_init__(self):
        xr = np.asarray(self.xr_km, dtype=float)
        if not (np.all(np.isfinite(xr)) and np.all(np.diff(xr) > 0)):
            raise ValueError("xr_km must be strictly increasing and finite")
        if len(self.coeffs) != len(xr) - 1:
            raise ValueError("coeffs length must be len(xr_km)-1")
        self.xr = xr
        self.RE_KM = 6371.0   # Earth radius in km; used for r/RE conversion in rhoYe_gcm3

    # def _layer_index(self, R_km: float) -> int:
    #     # right-edge binning; clamp to last layer
    #     k = int(np.searchsorted(self.xr, R_km, side="right") - 1)
    #     return max(0, min(k, len(self.xr) - 2))

    # def _Ye(self, r: float) -> float:
    #     if self.ye_fn is not None:
    #         return float(self.ye_fn(r))
    #     # default: core vs mantle split at r = 0.546 like classic PREM usage
    #     return self.Ye_core if r <= 0.546 else self.Ye_mantle

    # def rhoYe_gcm3(self, r_over_RE: float) -> float:
    #     r = float(np.clip(r_over_RE, 0.0, 1.0))
    #     R_km = r * self.RE_KM
    #     k = self._layer_index(R_km)
    #     a, b, c, d = self.coeffs[k]
    #     rho = a + b*r + c*r*r + d*r*r*r  # g/cm^3 (mass density)
    #     return self._Ye(r) * rho


    def _Ye(self, r):
        """Determine electron density Ye for given r; if ye_fn is given, it overrides the core/mantle split."""
        if self.ye_fn is not None:            
            Ye = self.ye_fn(r)   # must support arrays        
        else:            
            # default: core vs mantle split at r = 0.546 like classic PREM usage
            r_thr = 0.546
            Ye = np.where(r <= r_thr, self.Ye_core, self.Ye_mantle)

        return Ye
    

    def rhoYe_gcm3(self, r_over_RE):
        """Vectorised version of rhoYe_gcm3 for array inputs. Computes Ye*rho for each r/RE in the input array."""
        # Restrict to [0,1] and convert to km
        r = np.clip(r_over_RE, 0.0, 1.0)
        R_km = r * self.RE_KM

        # vectorised layer index calculation; right-edge binning; clamp to last layer
        k = np.searchsorted(self.xr, R_km, side="right") - 1
        k = np.clip(k, 0, len(self.xr) - 2)

        # Get polynomial coefficients for each r; shape (len(r), 4)
        coeffs = np.array(self.coeffs)
        a = coeffs[k, 0]
        b = coeffs[k, 1]
        c = coeffs[k, 2]
        d = coeffs[k, 3]

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

    
    def rhoYe_gcm3(self, r_over_RE):        
        """
        np.interp already supports arrays. No scalar casting.     
        """        
        r = np.clip(r_over_RE, 0.0, 1.0) 
        # simple, safe interpolation; replace with scipy PchipInterpolator if you like       
        return np.interp(r, self.r, self.y)
    


# 4) Pre-defined PREM --------------------------------------------
        
xr_km_PREM = [0., 1221.5, 3480.0, 5701.0, 5771.0, 5971.0, 6151.0,
               6346.6, 6356.0, 6368.0, 6371.0, 6371.0 + 15.0]  # PREM boundaries
coeffs_PREM = [
    (13.0885,  0.0,    -8.8381,  0.0),     # layer 1: a,b,c,d in r = R/RE
    (12.5815, -1.2638, -3.6426, -5.528),   # layer 2
    (7.9565,  -6.4761,  5.5283, -3.0807),  # ...
    (5.3197,  -1.4836,  0.0,      0.0),
    (11.2494, -8.0298,  0.0,      0.0),
    (7.1089,  -3.8045,  0.0,      0.0),
    (2.6910,   0.6924,  0.0,      0.0),
    (2.9,      0.0,     0.0,      0.0),
    (2.6,      0.0,     0.0,      0.0),
    (1.02,     0.0,     0.0,      0.0),
    (0.000,    0.0,     0.0,      0.0),
]

# Predefined PREM model instance
PREMmodel = LayeredPolyEarth(xr_km=xr_km_PREM, coeffs=coeffs_PREM, Ye_core=0.466, Ye_mantle=0.494)
















class EarthProbEvolve:
    """
    Evolve neutrino density matrix through Earth matter for 
    given oscillation parameters, NSI model, and Earth density profile.
    """
    def __init__(self,  model, osc_params=osc.osc_params_best, 
                 earthmodel: Optional[EarthModel] = None, Nst: int = 50, Nav: int = 50):
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
    
    def _precompute_slices(self, ceta: float):
        """
        Precompute slice geometry and density averages for a given ceta.
        These depend only on the trajectory, not on energy.
        Returns t1, step_sizes (Nst,), Vav (Nst,).
        """
        t1 = (self.ERad * ceta
            + np.sqrt(max(0.0, self.tRad**2 - self.ERad**2 * (1 - ceta**2)))
            ) / self.tRad

        k      = np.arange(self.Nst)
        xmins  = t1 * k       / self.Nst          # (Nst,)
        xmaxs  = t1 * (k + 1) / self.Nst          # (Nst,)
        dxs    = xmaxs - xmins                     # (Nst,)

        # Sample points inside each slice: shape (Nst, Nav+1)
        l  = np.arange(self.Nav + 1)
        xi = xmins[:, None] + dxs[:, None] * l[None, :] / (self.Nav + 1)

        # Vectorised r/RE — inline the geometry formula to avoid a Python loop
        norm        = 1.0 / (RE_KM + ATM_KM)
        root_common = np.sqrt(max(0.0, 1.0 - (norm * RE_KM)**2 * (1 - ceta**2)))
        r_over_RE   = (np.sqrt(np.maximum(0.0, 1 + xi**2 - 2 * xi * root_common))
                    / (norm * RE_KM))           # (Nst, Nav+1)

        # rhoYe evaluation — earthmodel must accepts ndarray input and return array output for this to be efficient
        rhoYe = self.earthmodel.rhoYe_gcm3(r_over_RE)   # (Nst, Nav+1)
        Vav   = rhoYe.mean(axis=1)                      # (Nst,)

        return t1, dxs, Vav


    def S_matrix_batch(self, enus_GeV: np.ndarray, ceta: float) -> np.ndarray:
        """
        Compute the (N, 3, 3) array of S-matrices for N energies at fixed ceta.
        Replaces the per-energy loop over S_matrix().
        """
        enus = np.asarray(enus_GeV, dtype=float)
        N    = len(enus)

        if np.arccos(ceta) >= np.pi / 2:
            return np.broadcast_to(np.eye(3, dtype=np.complex128), (N, 3, 3)).copy()

        _, dxs, Vav = self._precompute_slices(ceta)    # (Nst,), (Nst,)

        # ------------------------------------------------------------------ #
        # Build Hv(E) for all N energies at once
        # Hv[n] = U @ diag(0, dm12/2E, dm31/2E) @ U†
        # ------------------------------------------------------------------ #
        U    = osc.UPMNS(self.osc_params)
        dm12 = self.osc_params.delta_m12
        dm31 = self.osc_params.delta_m31

        diag_vals          = np.zeros((N, 3))
        diag_vals[:, 1]    = dm12 / (2 * enus)
        diag_vals[:, 2]    = dm31 / (2 * enus)

        # Hv[n,i,j] = Σ_a  U[i,a] · diag_vals[n,a] · U*[j,a]
        Hv = np.einsum('ia,na,ja->nij', U, diag_vals, U.conj())   # (N, 3, 3)

        # ------------------------------------------------------------------ #
        # Full Hamiltonian: H[n,k] = Hv[n] + s2GFNa·Vav[k]·Vf
        # shape: (N, Nst, 3, 3)
        # ------------------------------------------------------------------ #
        Vf = self._V_matrix()                                        # (3, 3)
        H  = (Hv[:, None, :, :]
            + (s2GFNa * Vav)[None, :, None, None] * Vf[None, None, :, :])

        # ------------------------------------------------------------------ #
        # Batched eigendecomposition — one call for all N·Nst matrices
        # ------------------------------------------------------------------ #
        Em, Um = np.linalg.eigh(H.reshape(N * self.Nst, 3, 3))
        Em = Em.reshape(N, self.Nst, 3)          # eigenvalues
        Um = Um.reshape(N, self.Nst, 3, 3)       # eigenvectors (columns)

        # ------------------------------------------------------------------ #
        # Phase factors: exp(-i · Em · tRad · dx_k)
        # ------------------------------------------------------------------ #
        phase = np.exp(-1j * Em * self.tRad * dxs[None, :, None])   # (N, Nst, 3)

        # Transfer matrices: M[n,k] = Um[n,k] @ diag(phase[n,k]) @ Um†[n,k]
        # M[n,k,i,j] = Σ_a  Um[n,k,i,a] · phase[n,k,a] · Um*[n,k,j,a]
        M = np.einsum('nkia,nka,nkja->nkij', Um, phase, Um.conj())  # (N, Nst, 3, 3)

        # ------------------------------------------------------------------ #
        # Sequential product over slices — only Nst iterations, each a
        # batched (N, 3, 3) matmul
        # ------------------------------------------------------------------ #
        S = np.eye(3, dtype=np.complex128)[None].repeat(N, axis=0)   # (N, 3, 3)
        for k in range(self.Nst):
            S = np.matmul(S, M[:, k])

        return S                                                      # (N, 3, 3)


    def evolve_rhosolar(self, rho_solar, enus_GeV, ceta):
        """
        rho_solar : (N, 3, 3) complex
        enus_GeV  : (N,) energies
        ceta      : scalar cos(nadir)
        Returns rho_earth : (N, 3, 3)
        """
        S    = self.S_matrix_batch(np.asarray(enus_GeV), ceta)   # (N, 3, 3)
        Sdag = np.swapaxes(S.conj(), -1, -2)
        return np.matmul(np.matmul(S, rho_solar), Sdag)


class EarthProbEvolveOld:
    def __init__(self,  model, osc_params=osc.osc_params_best, 
                 earthmodel: Optional[EarthModel] = None, Nst: int = 50, Nav: int = 50):
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
    
    def evolve_rhosolar(self, rho_solar, enus_GeV, ceta):
        """
        rho_solar_stack: (N,3,3) complex array  [your solar density matrices at Earth surface]
        enus_GeV       : (N,) energies
        ceta           : scalar cos(nadir)
        propagator     : an object with S_matrix(E, ceta) -> (3,3) complex

        Returns:
        rho_earth_stack : (N,3,3) complex
        """
        # 1) build S(E) for all energies
        S_list = [self.S_matrix(E, ceta) for E in enus_GeV]
        S = np.stack(S_list, axis=0)                    # (N,3,3)
        Sdag = np.swapaxes(S.conj(), -1, -2)            # (N,3,3)

        # 2) batch multiply: S ρ S†   (use matmul with batch dims)
        tmp = np.matmul(S, rho_solar)             # (N,3,3)
        rho_earth = np.matmul(tmp, Sdag)                # (N,3,3)


        return rho_earth 


        
