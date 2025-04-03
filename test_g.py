#import main as mp
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from scipy import integrate

def f(x):
    return (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)

def tab_phi(c,n):
    """
    c tel que I = [-c,c], désigne la longueur du demi intervalle sur lequel on estime g
    n désigne le nombre de points, n impair
    """
    h = c/((n-1)/2)
    T = np.zeros(n)
    s = int((n-1)/2)
    T[s] = 0.5
    for k in range(s+1):
        print(k*h)
        T[s+k] = integrate.quad(f, -np.inf, k*h)[0]
        T[s-k] = integrate.quad(f, -np.inf, -k*h)[0]
    return T

tab_phi(5,10000)
    
