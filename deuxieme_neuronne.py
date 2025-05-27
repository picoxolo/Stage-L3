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

def noyau_gaussien(sigma):
    k = np.zeros((31,31))
    for i in range(31):
        for j in range(31):
            k[i,j] = 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-((i-5)**2+(j-5)**2)/(2*sigma**2))
    return(k)

def noyau_gaussien_cached(sigma):
    return noyau_gaussien(sigma)

#print(mp.norme2(noyau_gaussien(0.40)) )

def fft_lamda_droite_sigma (sigma_1,sigma_2,a):
    return((np.abs(fft2(noyau_gaussien_cached(sigma_2)))** 2) *  fft2(rtg.F_vect(a,mp.autocor(noyau_gaussien_cached(sigma_1)))) - 1)

def fft_delta_droite_sigma(sigma_1,sigma_2,a):
    return(fft2(-1*noyau_gaussien_cached(sigma_2)) *
           fft2(noyau_gaussien_cached(sigma_2)*noyau_gaussien_cached(sigma_2)) *
           fft2(rtg.F_vect(a,mp.autocor(noyau_gaussien_cached(sigma_1)))))

    
def opti_sigma(Z, a):
    M, N = Z.shape
    def optimisation(x):
        half = len(x) // 2
        sigma_1 = x[0]
        sigma_2 = x[1]
        return (
            mp.norme2_2(fft_lambda(Z, a) - fft_lamda_droite_sigma(sigma_1, sigma_2, a))
            + mp.norme2_2(fft_delta(Z) - fft_delta_droite_sigma(sigma_1, sigma_2, a))
        )
    # Point de départ (vecteurs k_2 et k_1 mis bout à bout)
    x0 = np.array([1,1])
    # Optimisation
    result = minimize(optimisation, x0, options={'disp': True}, bounds = [(0.5,10),(0.5,10)]) 
    sigma_1_opt = result.x[0]
    sigma_2_opt = result.x[1]
    print(sigma_1_opt, sigma_2_opt)
    s1 = np.linspace(0.5, 10, 30)
    s2 = np.linspace(0.5, 10, 30)
    Z_vals = np.zeros((30, 30))
    for i in range(30):
        for j in range(30):
            Z_vals[i, j] = optimisation([s1[i], s2[j]])
        print(i)
    plt.contourf(s1, s2, Z_vals.T, levels=30)
    plt.xlabel("sigma_1")
    plt.ylabel("sigma_2")
    plt.colorbar(label="Loss_function")
    plt.title("Paysage de la fonction à minimiser")
    plt.show()
    return sigma_1_opt, sigma_2_opt, result.fun  # tu peux aussi retourner result si tu veux plus d’infos    

print(opti_sigma(mp.convol(noyau_gaussien(1),exg.seuil_vect(mp.convol(noyau_gaussien(1),mp.whitenoise(31,31,1)))),0))
    
    
    
