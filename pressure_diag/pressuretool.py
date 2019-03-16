import numpy as np

class pressure_diag():
    def __init__(self,statsdata,alpha_b=1.0/3,alpha_d=0.375,r_d=500):
        self.alpha_b = alpha_b
        self.alpha_d = alpha_d
        self.r_d = r_d
        self.t = [it.data.tolist() for it in statsdata.groups['timeseries']['t']]
        self.z = [iz.data.tolist() for iz in statsdata.groups['reference']['z']]
        self.updraft_dyn_pressure = statsdata.groups['profiles']['updraft_dyn_pressure']
        self.updraft_fraction = statsdata.groups['profiles']['updraft_fraction']
        self.updraft_w = statsdata.groups['profiles']['updraft_w']
        self.env_w = statsdata.groups['profiles']['env_w']
        self.updraft_b = statsdata.groups['profiles']['updraft_b']
        self.rho0 = statsdata.groups['reference']['rho0']

    def mean_pz_sink(self):
        avgp_z = np.apply_along_axis( np.gradient, 1, self.updraft_dyn_pressure, z )
        a_z = np.apply_along_axis( np.gradient, 1, self.updraft_fraction, z )
        return -(self.updraft_fraction * avgp_z - self.updraft_dyn_pressure * a_z)

    def pressure_drag(self):
        return -(self.updraft_w[:]-self.env_w[:])*abs(self.updraft_w[:]-self.env_w[:])*self.alpha_d*\
               self.rho0*np.sqrt(self.updraft_fraction[:])/self.r_d

    def pressure_buoy(self):
        return -self.alpha_b*self.rho0[:]*self.updraft_fraction[:]*self.updraft_b[:]
