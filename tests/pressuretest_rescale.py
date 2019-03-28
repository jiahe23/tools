import numpy as np
import netCDF4 as nc
import pandas as pd
from scipy import interpolate as intp

import sys
sys.path.insert(0,'/home/jiahe/CliMA/tools/LES_diag_tool/')
from updraftvar import vertical_rescale

'''
test data from Bomex.test4
select column results from several snapshots
[-1, -61, -121] for now
getting cloud and updraft related info and write into dataframe
'''
testdata = nc.Dataset('/export/data1/jiahe/LESdata/Output.Bomex.test4/stats/Stats.Bomex.Restart_2.nc')
cloud_fraction = testdata.groups['profiles']['cloud_fraction'][:].data
updraft_fraction = testdata.groups['profiles']['updraft_fraction'][:].data
cloud_pres = testdata.groups['profiles']['dyn_pressure_cloud'][:].data
cloud_buoy = testdata.groups['profiles']['buoyancy_cloud'][:].data - testdata.groups['profiles']['buoyancy_mean'][:].data
cloud_w = testdata.groups['profiles']['w_cloud'][:].data
updraft_pres = testdata.groups['profiles']['updraft_dyn_pressure'][:].data
updraft_buoy = testdata.groups['profiles']['updraft_b'][:].data - testdata.groups['profiles']['buoyancy_mean'][:].data
updraft_w = testdata.groups['profiles']['updraft_w'][:].data

dataout = pd.DataFrame(testdata.groups['reference']['z'][:].data,columns=['z'])
dataout['cloud_fraction_1'] = cloud_fraction[-1,:]
dataout['cloud_pres_1'] = cloud_pres[-1,:]
# dataout['cloud_buoy_1'] = cloud_buoy[-1,:]
# dataout['cloud_w_1'] = cloud_w[-1,:]
dataout['updraft_fraction_1'] = updraft_fraction[-1,:]
dataout['updraft_pres_1'] = updraft_pres[-1,:]

dataout.to_csv('./dataout.csv',index=False)

# testdata = nc.Dataset('/export/data1/jiahe/LESdata/Output.Bomex.test4/stats/Stats.Bomex.Restart_2.nc')
# z = testdata.groups['reference']['z'][:].data
#
# updraft = testdata.groups['profiles']['updraft_fraction'][:].data
# updraft_pressure = testdata.groups['profiles']['updraft_dyn_pressure'][:].data
#
# # call module
# vs = vertical_rescale(testdata)
# vs.set_updraft()
# vspress = vs.rescale_var(testdata.groups['profiles']['updraft_dyn_pressure'][:].data)
#
# # manual interpolation
# mpress = np.zeros_like(vspress)
# for irow in np.arange(mpress.shape[0]):
#     ifrac = updraft[irow,:]
#     idx = ifrac > 1e-3
#     ifrac = ifrac[idx]
#     idata = updraft_pressure[irow,idx]
#     iz = z[idx]
#     if idx.sum() == 0:
#         continue
#     elif idx.sum() == 1:
#         mpress[irow,0] = idata[0]
#         continue
#     iz = [1+float(i)/(len(iz)-1) for i in np.arange(len(iz))]
#     # print iz
#     f = intp.interp1d(iz,idata)
#     mpress[irow,:-1] = f(np.arange(1,2.0,0.01))
#     mpress[irow,-1] = idata[-1]
#
# print abs(vspress - mpress)<1e-3
#
# print vspress[1,0]
# print mpress[1,0]
