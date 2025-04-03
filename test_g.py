#import main as mp
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from scipy import integrate

def f(x):
    return (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)

def tab_phi(l,n):
    """
    l désigne la longueur de l'itervalle sur laquelle on estime l'image de g
    intervalle centré en 0
    n désigne le nombre de points, n impair
    """
    a, k = 0, 0
    h = l/n
    T = np.zeros(n)
    s = int((n-1)/2)
    T[s] = 0.5
    while s+k < n-1:
        a += h
        k += 1
        T[s - k] = T[s - k + 1] - integrate.quad(f, a-h, a)[0]
        T[s + k] = T[s + k - 1] + integrate.quad(f, a, a+h)[0]
    return T

tab_phi(4,9)
    
