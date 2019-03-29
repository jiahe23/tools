import numpy as np
import netCDF4 as nc
import sys
sys.path.insert(0,'/home/jiahe/CliMA/tools/LES_diag_tool/')
from updraftvar import updraft_analysis

testdata = nc.Dataset('/export/data1/jiahe/LESdata/Output.Bomex.test4/stats/Stats.Bomex.Restart_2.nc')
t = testdata.groups['timeseries']['t'][:].data
tidx = np.arange(-1,-11,-1)
upa = updraft_analysis(testdata)

output1 = upa.profile_timeave('updraft_dyn_pressure',tidx)
print output1
# print tidx
# print t[tidx]
output2, data, idx = upa.movingavg_time(testdata.groups['profiles']['updraft_dyn_pressure'][:].data,600)
# print idx
# print t[idx]
# print output2['t']
print output2['data_movingavg'][-1,:]
print abs(output1 - output2['data_movingavg'][-1,:]) < 1e-6
