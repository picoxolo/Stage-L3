# -*- coding: utf-8 -*-
"""
Created on Fri May 23 15:36:09 2025

@author: Astrid
"""
import matplotlib.pyplot as plt
import numpy as np
import main as mp
from numpy.fft import fft2, ifft2, fftfreq

from ymscript import laplacian as lap    # five-point laplacian (periodic boundary)
from ymscript import gauss      as G    # gaussian semigroup
from ymscript import riesz      as R    # riesz semigroup
from ymscript import randu      as U    # uniform samples
from ymscript import randg      as N    # normal samples
from ymscript import ntiply             # zoom-in by pixel repetition

'''
x = N((400,400))     # x = image de bruit blanc 400x400
y = G(x, 6)       # y = flou gaussien de x avec σ=6
z = -lap(y)           # z = laplacien de y
w = z*(z>0)           # w = "ReLU" de z
'''

def histo(U,n): #n:nombre de points
    sorted_array = U.flatten()
    plt.figure()
    plt.hist(sorted_array,cumulative = False, bins = 20, density = False, edgecolor = 'black')
    plt.title('Histogramme')

def kernel_freq(s, σ, p, q):
    """ construit un noyau dans le domaine fréquentiel

        s: nom du noyau ("gauss", "riesz", "disk", "square", etc)
        σ: paramètre du noyau
        p,q : images avec les indices de fréquence
    """

    from numpy import exp, sinc, fabs, fmax
    from numpy import pi as π
    r2 = p**2 + q**2
    if s[0] == "g": return exp(-2 * π**2 * σ**2 * r2)             # gauss
    if s[0] == "d": return sinc(2 * σ * r2**0.5)                  # disk
    if s[0] == "s": return sinc(2*σ*fabs(p)) * sinc(2*σ*fabs(q))  # square
    if s[0] == "r":                                               # riesz
        r2[0,0] = 1
        F = r2**(-σ/2)
        F[0,0] = 0
        return F
    if s[0] == "x": return 1j*p                                   # d/dx
    if s[0] == "y": return 1j*q                                   # d/dy
    if s[0] == "l": return -(p**2 + q**2)                       # laplacien
    return 1+0*p  #dirac


def filter(x, k, σ=3):
    """ filtre une image par un noyau

        x: image à niveau de gris
        k: nom du noyau ("gauss", "riesz", "disk", "square", etc)
        σ: paramètre du noyau
    """
    h,w = x.shape                           # shape of the rectangle
    p,q = np.meshgrid(fftfreq(w), fftfreq(h))  # build frequency abscissae
    X = fft2(x)                             # move to the frequency domain
    F = kernel_freq(k, σ, p, q)             # filter in the frequency domain
    Y = F*X                                 # apply filter
    y = ifft2(Y).real                       # go back to the spatial domain
    return y

def filter_bis(x,k,σ=3): #k noyau fréquentiel
    #h,w = x.shape                           # shape of the rectangle
    #p,q = np.meshgrid(fftfreq(w), fftfreq(h))  # build frequency abscissae
    X = fft2(x)                             # move to the frequency domain          # filter in the frequency domain
    Y = k*X                                 # apply filter
    y = ifft2(Y).real                       # go back to the spatial domain
    return y

def seuil(x):
    if x>0:
        return(1)
    return(0)
seuil_vect = np.vectorize(seuil)

def ReLu(x,a,b):
    def temp(x):
        if x > a:
            return b*(x-a)
        return (0)
    return np.vectorize(temp)

def seuil_pct(u,p):
    M,N = u.shape
    L = np.sort(u.flatten())
    s = L[int(np.floor(p*M*N))]
    v  = np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            if u[i,j] >= s:
                v[i,j] = u[i,j]
    return v
            



#%%

W = mp.whitenoise(800,800,1)
u = filter(W,"gauss", 10)
v = seuil_pct(u,0.7)
z = filter(v, "gauss", 3)
z = filter(z,"lap")
w = seuil_pct(z,0.9)

mp.printimage([u,v, z, w], [ 'u', 'v', 'z', 'w'])












































'''
def kernel(s,σ, M,N):
    from numpy import exp, sinc, fabs, fmax
    from numpy import pi as π
    img = np.zeros((M,N))
    if s[0] == "g":
        for p in range(M):
            for q in range(N):
                r2 = p**2 + q**2
                img[p,q] = exp(-2 * π**2 * σ**2 * r2)             # gauss
    if s[0] == "d":
        for p in range(M):
            for q in range(N):
                r2 = p**2 + q**2
                img[p,q] = sinc(2 * σ * r2**0.5)                  # disk
    if s[0] == "s":
        for p in range(M):
            for q in range(N):
                r2 = p**2 + q**2
                img[p,q] = sinc(2*σ*fabs(p)) * sinc(2*σ*fabs(q))  # square
    if s[0] == "r":  
        for p in range(M):
            for q in range(N):
                r2 = p**2 + q**2
                img[p,q] =                                              # riesz
                r2[0,0] = 1
                F = r2**(-σ/2)
                F[0,0] = 0
                return F
    if s[0] == "x":
        for p in range(M):
            for q in range(N):
                img[p,q] =  1j*p                                   # d/dx
    if s[0] == "x":
        for p in range(M):
            for q in range(N):
                img[p,q] =  1j*q                                   # d/dy
    if s[0] == "lap":
        for p in range(M):
            for q in range(N):
                img[p,q] = -(p**2 + q**2)                       # laplacien
     if s[0] == "dirac":
         img[0,0] = 1
'''

