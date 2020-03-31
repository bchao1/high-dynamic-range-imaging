import numpy as np 
from matplotlib import pyplot as plt 

def linear_hat(zmin = 0, zmax = 255):
    zmid = zmin + (zmax - zmin) // 2
    def hat(z):
        return (z - zmin if z <= zmid else zmax - z) + 1
    return np.vectorize(hat)
    
def sin_hat(zmin = 0, zmax = 255):
    width = zmax - zmin
    zmid = zmin + (zmax - zmin) // 2
    def hat(z):
        return zmid * np.sin((z - zmin) * np.pi / width) + 1
    return hat

def gaussian_hat(zmin = 0, zmax = 255, sigma = 3):
    mean = zmin + (zmax - zmin) // 2
    radius = zmax - mean
    sigma = mean * 1.0 / sigma
    def hat(z):
        return mean * np.exp(-0.5 * ((z - mean) / sigma)**2) + 1
    return hat

if __name__ == '__main__':
    x = np.arange(256)
    plt.plot(x, linear_hat()(x), label = 'linear')
    plt.plot(x, gaussian_hat()(x), label = 'gaussian')
    plt.plot(x, sin_hat()(x), label = 'sin')
    plt.legend()
    plt.savefig('./images/hat.png')