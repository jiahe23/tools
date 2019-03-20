import numpy as np

class updraft_analysis():
    def __init__(self,statsdata):
        self.statsdata = statsdata

    def masked_by_updraft(self,var):
        if not isinstance(var, np.ndarray):
            sys.exit('Input Must Be np.ndarray!')
        if np.shape(var) != np.shape(self.statsdata.groups['profiles']['updraft_fraction']):
            sys.exit('ERROR: Input Dimension not Match Updraft')
        return np.ma.masked_where(self.statsdata.groups['profiles']['updraft_fraction'][:].data ==0,
                                var)

    def profile_timeave(self,varname,tidx):
        data = self.statsdata.groups['profiles'][varname][:].data
#        return self.masked_by_updraft(data)[tidx,:].mean(axis=0)
        return data.mean(axis=0)
