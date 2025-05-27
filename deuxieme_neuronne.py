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

def opti(Z, a):
    M, N = Z.shape
    def optimisation(x):
        # x contient k_2 et k_1 concaténés et aplatis
        half = len(x) // 2
        k_2 = x[:half].reshape((M, N))
        k_1 = x[half:].reshape((M, N))
        # Appel à tes fonctions avec les bons arguments
        return (
            mp.norme2_2(fft_lambda(Z, a) - fft_lamda_droite(k_2, k_1, a))
            + mp.norme2_2(fft_delta(Z) - fft_delta_droite(k_2, k_1, a))
        )
    # Point de départ (vecteurs k_2 et k_1 mis bout à bout)
    x0 = np.zeros(2 * M * N) + 0.1
    # Optimisation
    result = minimize(optimisation, x0, options={'disp': True, 'maxiter': 1})
    k_2_opt = result.x[:M * N].reshape((M, N))
    k_1_opt = result.x[M * N:].reshape((M, N))
    return k_2_opt, k_1_opt, result.fun  # tu peux aussi retourner result si tu veux plus d’infos

print(opti(exg.seuil_vect(mp.whitenoise(10,10,1)),0))
    
    
    
