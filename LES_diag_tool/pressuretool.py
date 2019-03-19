import numpy as np

class pycles_pressure_diag():
    def __init__(self,statsdata,alpha_b=1.0/3,alpha_d=0.375,r_d=500):
        self.alpha_b = alpha_b
        self.alpha_d = alpha_d
        self.r_d = r_d
        self.statsdata = statsdata

        self.t = [it.data.tolist() for it in statsdata.groups['timeseries']['t']]
        self.z = [iz.data.tolist() for iz in statsdata.groups['reference']['z']]

        # self.updraft_dyn_pressure = statsdata.groups['profiles']['updraft_dyn_pressure']
        self.updraft_fraction = statsdata.groups['profiles']['updraft_fraction']
        self.updraft_w = statsdata.groups['profiles']['updraft_w']
        self.env_w = statsdata.groups['profiles']['env_w']
        self.updraft_b = statsdata.groups['profiles']['updraft_b']
        self.buoyancy_mean = statsdata.groups['profiles']['buoyancy_mean']
        self.rho0 = statsdata.groups['reference']['rho0']

    def avgp_z(self):
        updraft_dyn_pressure = self.statsdata.groups['profiles']['updraft_dyn_pressure'][:].data
        return np.apply_along_axis(np.gradient, 1, updraft_dyn_pressure, self.z)

    def fraction_z(self):
        return np.apply_along_axis(np.gradient, 1, self.updraft_fraction, self.z)

    def mean_pz_sink(self):
        # one last term regarding the interface mean pressure is missing temporarily
        # the missing term slightly cancels the second term <updraft_press*area_gradient>
        return -( self.updraft_fraction * self.avgp_z() + self.updraft_dyn_pressure * self.fraction_z() )

    def pressure_drag(self):
        return -(self.updraft_w[:]-self.env_w[:])*abs(self.updraft_w[:]-self.env_w[:])*self.alpha_d*\
               self.rho0*np.sqrt(self.updraft_fraction[:])/self.r_d

    def pressure_buoy(self):
        return -self.alpha_b*self.rho0[:]*self.updraft_fraction[:]*(self.updraft_b[:]-self.buoyancy_mean[:])


