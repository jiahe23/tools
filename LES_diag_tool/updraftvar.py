import numpy as np
import pandas as pd
from pandas import DataFrame as df
from scipy import interpolate as intp
import sys

class updraft_analysis():
    def __init__(self,statsdata):
        self.statsdata = statsdata

    def masked_by_updraft(self,data,maskedby='updraft_fraction',maskthre=1e-6):
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

    def cloud_top_base(self):
        output = {}
        cloud_info = self.statsdata.groups['profiles']['cloud_fraction'][:].data>1e-6
        output['topidx'] = np.zeros(cloud_info.shape[0])
        output['baseidx'] = np.zeros(cloud_info.shape[0])
        output['heightcnt'] = np.zeros(cloud_info.shape[0])
        for it in np.arange(cloud_info.shape[0]):
            output['heightcnt'][it] = cloud_info[it,:].sum()
            for iz in np.arange(cloud_info.shape[1]):
                if cloud_info[it,iz]:
                    output['baseidx'][it] = iz
                    break
            output['topidx'][it] = output['baseidx'][it]+output['heightcnt'][it]-1
        return output

    def updraft_top_base(self):
        output = {}
        updraft_info = self.statsdata.groups['profiles']['updraft_fraction'][:].data>1e-6
        output['topidx'] = np.zeros(updraft_info.shape[0])
        output['baseidx'] = np.zeros(updraft_info.shape[0])
        output['heightcnt'] = np.zeros(updraft_info.shape[0])
        for it in np.arange(updraft_info.shape[0]):
            output['heightcnt'][it] = updraft_info[it,:].sum()
            for iz in np.arange(updraft_info.shape[1]):
                if updraft_info[it,iz]:
                    output['baseidx'][it] = iz
                    break
            output['topidx'][it] = output['baseidx'][it]+output['heightcnt'][it]-1
        return output

    def movingavg_time(self,data,tw,label='updraft_fraction'):
        '''
        tw is the moving average time window in seconds
        label is updraft_fraction as default and can be changed to cloud_fraction
        '''
        if not isinstance(data, np.ndarray):
            sys.exit('Input Must Be np.ndarray!')
        output = {}
        t = self.statsdata.groups['timeseries']['t'][:].data
        data = self.masked_by_updraft(data,label)
        T_cnt = int(np.floor_divide(t.max(),tw))
        output['data_movingavg'] = np.zeros((T_cnt,data.shape[1]))
        output['t'] = np.zeros(T_cnt)
        for iT in np.arange(len(output['t'])):
            # output['t'] archive the end time of the miving avg (output['t']-tw,output['t']]
            output['t'][iT] = t[0] + (iT+1)*tw
            idx = [it for it in np.arange(len(t)) if ( t[it] > t[0]+iT*tw and t[it] <= t[0]+(iT+1)*tw )]
            output['data_movingavg'][iT] = data[idx,:].mean(axis=0)
        return output

    def data2frame(self,data,z,t,maskedby):
        if not isinstance(data, np.ndarray):
            sys.exit('Input Data Must Be np.ndarray!')
        if not (data.shape==maskedby.shape):
            sys.exit('Input Data and Mask Must Match Dimensions!')
        zz,tt = np.meshgrid(z,t)
        if not (zz.shape==data.shape):
            sys.exit('Input Data Must Match Dimensions with z & t!')
        output = df(data.reshape(len(z)*len(t),1), columns=['data'])
        output['mask'] = maskedby.reshape(len(z)*len(t),1)
        output['z'] = zz.reshape(len(z)*len(t),1)
        output['t'] = tt.reshape(len(z)*len(t),1)
        return output


class vertical_rescale():
    def __init__(self,statsdata):
        self.zscale = np.arange(1,2.01,0.01)
        self.statsdata = statsdata
        self.t = statsdata.groups['timeseries']['t'][:].data
        self.z = statsdata.groups['reference']['z'][:].data

    def set_cloud(self):
        self.fraction_mask = self.statsdata.groups['profiles']['cloud_fraction'][:].data

    def set_updraft(self):
        self.fraction_mask = self.statsdata.groups['profiles']['updraft_fraction'][:].data

    def __norm_z(self):
        output = np.zeros_like(self.fraction_mask,dtype='float32')
        for irow in np.arange(output.shape[0]):
            idx = [i for i in np.arange( self.z.shape[0] ) if self.fraction_mask[irow,i]>1e-6]
            if len(idx)==1:
                output[irow,idx[0]] = 1
            elif len(idx)>1:
                output[irow,idx] = 1+np.arange(len(idx))*(self.zscale[-1]-self.zscale[0])/(len(idx)-1)
        return output

    def rescale_var(self,vardata):
        z_norm = self.__norm_z()
        output = np.zeros((len(self.t), self.zscale.shape[0]))
        # print output.shape
        # print self.zscale.shape
        for irow in np.arange(output.shape[0]):
            iz = z_norm[irow,:]
            idata = vardata[irow,:]
            idx = iz != 0
            if idx.sum() == 1:
                output[irow,0] = idata[0]
            elif idx.sum() == 0:
                continue
            else:
                f = intp.interp1d(iz[idx], idata[idx])
                output[irow,:-1] = f(self.zscale[:-1])
                output[irow,-1] = idata[idx][-1]
        return output
