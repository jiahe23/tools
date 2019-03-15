import numpy as np

def pressure_diag():
    def __init__(self):
        self.z = z
        self.updraft



def mean_pz(z, updraft_dyn_pressure, updraft_fraction):
    avgp_z = np.apply_along_axis( np.gradient, 1, updraft_dyn_pressure, z )
    a_z = np.apply_along_axis( np.gradient, 1, updraft_fraction, z )
    return updraft_fraction * avgp_z - updraft_dyn_pressure * a_z

def pressure_drag():


def pressure_buoy():


