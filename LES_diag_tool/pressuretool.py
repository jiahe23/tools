import numpy as np

class pycles_pressure_diag():
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
        return np.apply_along_axis(np.gradient, 1, self.updraft_dyn_pressure, self.z)

    def DaiDz(self):
        return np.apply_along_axis(np.gradient, 1, self.updraft_fraction, self.z)

    def mean_pz_sink(self):
        # one last term regarding the interface mean pressure is missing temporarily
        # the missing term slightly cancels the second term <updraft_press*area_gradient>
        output = {}
        output['dpidz'] = -self.DpiDz()
        tmp_ai = self.updraft_fraction
        tmp_ai[tmp_ai==0] = 1e-6
        output['ai_contribution'] = -self.updraft_dyn_pressure * self.DaiDz()/tmp_ai
        output['mean_sink'] = -(self.updraft_fraction*self.DpiDz() + self.updraft_dyn_pressure*self.DaiDz())/tmp_ai
        return output

    def pressure_drag(self):
        return -(self.updraft_w-self.env_w)*abs(self.updraft_w-self.env_w)*self.alpha_d*\
               self.rho0*np.sqrt(self.updraft_fraction)/self.r_d

    def pressure_buoy(self):
        return -self.alpha_b*self.rho0*self.updraft_fraction*self.updraft_reletive_buoyancy


