import numpy as np
from updraftvar import updraft_analysis as upa
from scipy import interpolate.interp1d

class pycles_pressure_decompose():
    def __init__(self,statsdata,alpha_b=1.0/3,alpha_d=0.375,r_d=500):
        self.alpha_b = alpha_b
        self.alpha_d = alpha_d
        self.r_d = r_d
        self.statsdata = statsdata

        self.t = [it.data.tolist() for it in statsdata.groups['timeseries']['t']]
        self.z = [iz.data.tolist() for iz in statsdata.groups['reference']['z']]

        self.updraft_dyn_pressure = statsdata.groups['profiles']['updraft_dyn_pressure'][:].data
        self.updraft_fraction = statsdata.groups['profiles']['updraft_fraction'][:].data
        self.updraft_w = statsdata.groups['profiles']['updraft_w'][:].data
        self.env_w = statsdata.groups['profiles']['env_w'][:].data
        self.updraft_reletive_buoyancy = statsdata.groups['profiles']['updraft_b'][:].data-\
                                         statsdata.groups['profiles']['buoyancy_mean'][:].data
        self.rho0 = statsdata.groups['reference']['rho0'][:].data

    def DpiDz(self):
        output = np.apply_along_axis(np.gradient, 1, self.updraft_dyn_pressure, self.z)
        return upa(self.statsdata).masked_by_updraft(output)

    def DaiDz(self):
        output = np.apply_along_axis(np.gradient, 1, self.updraft_fraction, self.z)
        return upa(self.statsdata).masked_by_updraft(output)

    def mean_dpdz(self):
        # one last term regarding the interface mean pressure is missing temporarily
        # the missing term slightly cancels the second term <updraft_press*area_gradient>
        output = {}
        output['dpidz'] = -self.DpiDz()
        tmp_ai = self.updraft_fraction
        tmp_ai[tmp_ai==0] = 1e-6
        output['ai_contr'] = -self.updraft_dyn_pressure * self.DaiDz()/tmp_ai
        output['mean_sink'] = -(self.updraft_fraction*self.DpiDz() + self.updraft_dyn_pressure*self.DaiDz())/tmp_ai
        return output

    def mean_pacceleration(self):
        output = {}
        output['pi_contr'] = -self.updraft_fraction*self.DpiDz()
        output['ai_contr'] = -self.updraft_dyn_pressure * self.DaiDz()
        output['total'] = output['pi_contr'] + output['ai_contr']
        return output

    def pressure_drag(self):
        return -(self.updraft_w-self.env_w)*abs(self.updraft_w-self.env_w)*self.alpha_d*\
               self.rho0*np.sqrt(self.updraft_fraction)/self.r_d

    def pressure_buoy(self):
        return -self.alpha_b*self.rho0*self.updraft_fraction*self.updraft_reletive_buoyancy

class pycles_pressure_para():
    alpha_b = np.nan
    alpha_d = np.nan
    r_d = np.nan

    def __init__(self,statsdata,dx=50,dy=50):
        self.statsdata = statsdata
        self.dx = dx
        self.dy = dy
        self.rho0 = statsdata.groups['reference']['rho0'][:].data
        self.updraft_fraction = statsdata.groups['profiles']['updraft_fraction'][:].data
        self.upa = upa(self.statsdata)

    def updraft_relative_b(self):
        output = self.statsdata.groups['profiles']['updraft_b'][:].data - \
                 self.upa.masked_by_updraft( self.statsdata.groups['profiles']['buoyancy_mean'][:].data )
        return self.upa.masked_by_updraft(output)

    def wdiff(self):
        output = self.statsdata.groups['profiles']['updraft_w'][:].data - \
                 self.statsdata.groups['profiles']['env_w'][:].data
        return self.upa.masked_by_updraft(output)

    def buoy_contr(self):
        output = {}
        output['dpdz'] = -self.rho0 * self.updraft_relative_b()
        output['dwdt'] = -self.rho0 * self.updraft_fraction * self.updraft_relative_b()
        for item in output:
            output[item] = self.upa.masked_by_updraft(output[item])
        return output

    def drag_contr(self):
        # wdiff = self.statsdata.groups['profiles']['updraft_w'][:].data - self.statsdata.groups['profiles']['env_w'][:].data
        output = {}
        tmpa = self.updraft_fraction
        tmpa[abs(self.updraft_fraction)<1e-6] = 1e-6
        output['dpdz'] = -self.rho0 * self.wdiff() * abs(self.wdiff()) / np.sqrt( tmpa )/np.sqrt( self.dx * self.dy )
        output['dwdt'] = -self.rho0 * self.wdiff() * abs(self.wdiff()) * np.sqrt( self.updraft_fraction )/np.sqrt( self.dx * self.dy )
        for item in output:
            output[item] = self.upa.masked_by_updraft(output[item])
        return output
