import main as mp
import retrouver_g as rtg
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from scipy.integrate import tplquad
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def fft_lambda(Z,a):
    M,N = Z.shape()
    A = 1/(M*N) * mp.autocor(Z)
    p = 1 - rtg.phi(a)
    A= A - p^2
    return(fft2(A))

def fft_lamda_droite (k_2,k_1,a):
    return((np.abs(fft2(k_2)) ^2) *  fft2(rtg.F(a,mp.autocor(k_1))) - 1)

def fft_delta(Z):
    return(fft2(-1*Z)*fft2(Z*Z))

def fft_delta_droite(k_2,k_1):
    return(fft2(-1*k_2) * fft2(k_2*k_2) * fft2(rtg.F(a,mp.autocor(k_1))))

def optimisation(Z,k_2,k_1,a):
    return(mp.norme2(fft_lambda(Z,a)-fft_lamda_droite(k_2,k_1,a)) + mp.norme2(fft_delta(Z)-fft_delta_droite(k_2,k_1)))
    
    
    