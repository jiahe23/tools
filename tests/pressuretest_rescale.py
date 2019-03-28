import numpy as np
import netCDF4 as nc
from scipy import interpolate as intp

import sys
sys.path.insert(0,'/home/jiahe/CliMA/tools/LES_diag_tool/')
from updraftvar import vertical_rescale

testdata = nc.Dataset('/export/data1/jiahe/LESdata/Output.Bomex.test4/stats/Stats.Bomex.Restart_2.nc')
z = testdata.groups['reference']['z'][:].data

updraft = testdata.groups['profiles']['updraft_fraction'][:].data
updraft_pressure = testdata.groups['profiles']['updraft_dyn_pressure'][:].data

# call module
vs = vertical_rescale(testdata)
vs.set_updraft()
vspress = vs.rescale_var(testdata.groups['profiles']['updraft_dyn_pressure'][:].data)

# manual interpolation
mpress = np.zeros_like(vspress)
for irow in np.arange(mpress.shape[0]):
    ifrac = updraft[irow,:]
    idx = ifrac > 1e-3
    ifrac = ifrac[idx]
    idata = updraft_pressure[irow,idx]
    iz = z[idx]
    if idx.sum() == 0:
        continue
    elif idx.sum() == 1:
        mpress[irow,0] = idata[0]
        continue
    iz = [1+float(i)/(len(iz)-1) for i in np.arange(len(iz))]
    # print iz
    f = intp.interp1d(iz,idata)
    mpress[irow,:-1] = f(np.arange(1,2.0,0.01))
    mpress[irow,-1] = idata[-1]

print abs(vspress - mpress)<1e-3

print vspress[1,0]
print mpress[1,0]
