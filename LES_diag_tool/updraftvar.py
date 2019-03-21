import numpy as np

class updraft_analysis():
    def __init__(self,statsdata):
        self.statsdata = statsdata

    def masked_by_updraft(self,data,maskedby='updraft_fraction',maskthre=1e-3):
        if not isinstance(data, np.ndarray):
            sys.exit('Input Must Be np.ndarray!')
        if np.shape(data) != np.shape(self.statsdata.groups['profiles'][maskedby]):
            sys.exit('ERROR: Input Dimension not Match Mask')
        return np.ma.masked_where(self.statsdata.groups['profiles'][maskedby][:].data < maskthre,
                                  data)

    def profile_timeave(self,varname,tidx):
        maskvar = 'updraft_fraction'
        if varname[-5:] == 'cloud':
            maskvar = 'cloud_fraction'
        if varname[:5] == 'cloud':
            maskvar = 'cloud_fraction'
        data = self.statsdata.groups['profiles'][varname][:].data
        return self.masked_by_updraft(data,maskedby=maskvar)[tidx,:].mean(axis=0)

    def derived_timeave(self,data,tidx):
        if not isinstance(data, np.ndarray):
            sys.exit('Input Must Be np.ndarray!')
        return self.masked_by_updraft(data)[tidx,:].mean(axis=0)
