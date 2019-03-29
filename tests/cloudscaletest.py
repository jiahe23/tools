import numpy as np
import netCDF4 as nc
import sys
sys.path.insert(0,'/home/jiahe/CliMA/tools/LES_diag_tool/')
from updraftvar import updraft_analysis

'''
test data from Bomex.test4
select column results from several snapshots
[-1, -61, -121] for now
getting cloud and updraft related info and write into dataframe
'''
testdata = nc.Dataset('/export/data1/jiahe/LESdata/Output.Bomex.test4/stats/Stats.Bomex.Restart_2.nc')
cloud_info = updraft_analysis(testdata).cloud_top_base()
updraft_info = updraft_analysis(testdata).updraft_top_base()

print "=================== cloud =================="
print cloud_info['topidx'][-1]
print cloud_info['baseidx'][-1]
print cloud_info['heightcnt'][-1]

print "=================== updraft =================="
print updraft_info['topidx'][-1]
print updraft_info['baseidx'][-1]
print updraft_info['heightcnt'][-1]
