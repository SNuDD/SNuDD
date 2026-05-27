import numpy as np
from scipy import interpolate

from snudd.config import get_data

conv2ty = 365.25 * 1000
E_low_ext = 1e-2  # Energy to extend everything to in keVee

E_rn220_bkg, rn220_bkg = np.loadtxt(get_data("exps/lz/bkg_individual/Rn220.csv")).T
E_rn220_bkg = np.insert(E_rn220_bkg, 0, E_low_ext)
rn220_bkg = np.insert(rn220_bkg, 0, rn220_bkg[0])
E_rn220_bkg = np.insert(E_rn220_bkg, len(E_rn220_bkg), 100)
rn220_bkg = np.insert(rn220_bkg, len(rn220_bkg), rn220_bkg[-1]) * conv2ty

E_rn222_bkg, rn222_bkg = np.loadtxt(get_data("exps/lz/bkg_individual/Rn222.csv")).T
E_rn222_bkg = np.insert(E_rn222_bkg, 0, E_low_ext)
rn222_bkg = np.insert(rn222_bkg, 0, rn222_bkg[0])
E_rn222_bkg = np.insert(E_rn222_bkg, len(E_rn222_bkg), 100)
rn222_bkg = np.insert(rn222_bkg, len(rn222_bkg), rn222_bkg[-1]) * conv2ty

E_xe136_bkg, xe136_bkg = np.loadtxt(get_data("exps/lz/bkg_individual/Xe136.csv")).T
E_xe136_bkg = np.insert(E_xe136_bkg, 0, E_low_ext)
xe136_bkg = np.insert(xe136_bkg, 0, xe136_bkg[0]) * conv2ty  # Not quite right, but doesn't matter. Rn dominates at low Es

E_kr85_bkg, kr85_bkg = np.loadtxt(get_data("exps/lz/bkg_individual/Kr85.csv")).T
E_kr85_bkg = np.insert(E_kr85_bkg, 0, E_low_ext)
kr85_bkg = np.insert(kr85_bkg, 0, kr85_bkg[0])
E_kr85_bkg = np.insert(E_kr85_bkg, len(E_kr85_bkg), 100)
kr85_bkg = np.insert(kr85_bkg, len(kr85_bkg), kr85_bkg[-1]) * conv2ty

E_det_bkg, det_bkg = np.loadtxt(get_data("exps/lz/bkg_individual/DetSurEnv.csv")).T
E_det_bkg = np.insert(E_det_bkg, 0, E_low_ext)
det_bkg = np.insert(det_bkg, 0, det_bkg[0])
E_det_bkg = np.insert(E_det_bkg, len(E_det_bkg), 100)
det_bkg = np.insert(det_bkg, len(det_bkg), det_bkg[-1]) * conv2ty

rn220_bkg_interp = interpolate.interp1d(E_rn220_bkg/1e6, rn220_bkg, kind="linear")
rn222_bkg_interp = interpolate.interp1d(E_rn222_bkg/1e6, rn222_bkg, kind="linear")
xe136_bkg_interp = interpolate.interp1d(E_xe136_bkg/1e6, xe136_bkg, kind="linear")
kr85_bkg_interp = interpolate.interp1d(E_kr85_bkg/1e6, kr85_bkg, kind="linear")
det_bkg_interp = interpolate.interp1d(E_det_bkg/1e6, det_bkg, kind="linear")

### This is the one to import
total_bkg_spec_fn =  lambda E_R: rn220_bkg_interp(E_R) + \
                  rn222_bkg_interp(E_R) + \
                  xe136_bkg_interp(E_R) + \
                  kr85_bkg_interp(E_R) + \
                  det_bkg_interp(E_R)

