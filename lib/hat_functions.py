import numpy as np 
from matplotlib import pyplot as plt 

def linear_hat(zmin, zmax):
    zmid = zmin + (zmax - zmin) // 2
    def hat(z):
        return (z - zmin if z <= zmid else zmax - z) + 1
    return hat
    
def sin_hat(zmin, zmax):
    width = zmax - zmin
    def hat(z):
        return zmax * np.sin((z - zmin) * np.pi / width) + 1
    return hat

def gaussian_hat(zmin, zmax, sigma = 3):
    mean = zmin + (zmax - zmin) // 2
    radius = zmax - mean
    sigma = mean * 1.0 / sigma
    def hat(z):
        return zmax * np.exp(-0.5 * ((z - mean) / sigma)**2) + 1
    return hat