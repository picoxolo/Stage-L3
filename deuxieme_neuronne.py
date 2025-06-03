import main as mp
import exemples_k as exk
import retrouver_g as rtg
import exemples_g as exg
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from scipy.integrate import tplquad
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from functools import lru_cache


def fft_lambda(Z,a):
    M,N = np.shape(Z)
    A = 1/(M*N) * mp.autocor(Z)
    p = 1 - rtg.phi(a)
    A= A - p**2
    return(fft2(A))

def fft_lamda_droite (k_2,k_1,a):
    return((np.abs(fft2(k_2))** 2) *  fft2(rtg.F_vect(a,mp.autocor(k_1))) - 1)

def fft_delta(Z):
    M,N = np.shape(Z)
    return(1/(M*N) * fft2(-1*Z)*fft2(Z*Z))

def fft_delta_droite(k_2,k_1,a):
    return(fft2(-1*k_2) * fft2(k_2*k_2) * fft2(rtg.F_vect(a,mp.autocor(k_1))))

def optiv0(Z,k_2,k_1,a):
    M,N = np.shape(Z)
    def optimisation(k_2,k_1):
        return(mp.norme2_2(fft_lambda(Z,a)-fft_lamda_droite(k_2.reshape((M,N)),k_1.reshape((M,N)),a))
               + mp.norme2_2(fft_delta(Z)-fft_delta_droite(k_2.reshape((M,N)),k_1.reshape((M,N)),a)))
    return minimize(optimisation,np.zeros((M,N)).flatten()@np.zeros((M,N)).flatten())


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

#print(opti(exg.seuil_vect(mp.whitenoise(10,10,1)),0))


def noyau_gaussien(sigma):
    k = np.zeros((101,101))
    for i in range(101):
        for j in range(101):
            k[i,j] = 1/(2*np.pi*sigma**2) * np.exp(-((i-50)**2+(j-50)**2)/(2*sigma**2))
    return(k)

def noyau_gaussien_cached(sigma):
    return noyau_gaussien(sigma)

def noyau_gaussien_prime(sigma):
    k = noyau_gaussien_cached(sigma)
    M, N = k.shape
    x = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            x[i, j] = i**2 + j**2
    return k * x

def noyau_gaussien_prime_cached(sigma):
    return noyau_gaussien_prime(sigma)

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
        ),
    # Point de départ (vecteurs k_2 et k_1 mis bout à bout)
    x0 = np.array([1,1])
    # Optimisation
    result = minimize(optimisation, x0, options={'disp': True}, bounds = [(0.5,10),(0.5,10)]) 
    sigma_1_opt = result.x[0]
    sigma_2_opt = result.x[1]
    print(sigma_1_opt, sigma_2_opt)
    s1 = np.linspace(0.5, 10, 5)
    s2 = np.linspace(0.5, 10, 5)
    Z_vals = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            Z_vals[i, j] = optimisation([s1[i], s2[j]])
        print(i)
    plt.contourf(s1, s2, Z_vals.T, levels=5)
    plt.xlabel("sigma_1")
    plt.ylabel("sigma_2")
    plt.colorbar(label="Loss_function")
    plt.title("Paysage de la fonction à minimiser")
    plt.show()
    return sigma_1_opt, sigma_2_opt, result.fun  # retourner result pour plus d'info    

#print(opti_sigma(mp.convol(noyau_gaussien(1),exg.seuil_vect(mp.convol(noyau_gaussien(1),mp.whitenoise(101,101,1)))),0))


def energie(Z,k,delta):
    return(mp.norme2_2(ifft2(fft2(Z)/(fft2(k)+delta)) - (ifft2(fft2(Z)/(fft2(k)+delta)))**2))

k = np.zeros((100,100))
k[0,0] = 1

k_ini = np.ones((100,100))

k_2 = np.zeros((100,100))
k_2[0,0] = 1
k_2[0,1] = 1

k_3 = np.zeros((100,100))
k_3[0,0] = 1
k_3[0,1] = 1
k_3[0,2] = 1
k_3[1,1] = 1
k_3[1,0] = 1
k_3[1,2] = 1
k_3[2,1] = 1
k_3[2,0] = 1
k_3[2,2] = 1
#print(energie(exg.seuil_vect(mp.whitenoise(100,100,1)),k,0.000001))

def grad_e(Z,k,delta):
    M,N = np.shape(Z)
    return(2/(M*N) * 
        mp.convol((ifft2(fft2(Z)/(fft2(k)+delta)) - (ifft2(fft2(Z)/(fft2(k)+delta)))**2) * (2*ifft2(fft2(Z)/(fft2(k)+delta))- 1),
           fft2(fft2(Z)/(fft2(k)**2+delta))))

#mp.printimage([grad_e(exg.seuil_vect(mp.whitenoise(100,100,1)),k_ini,0.000001)],["grad"])

def descente(Z,k,delta,eps):
    for i in range(10000):
        grad = grad_e(Z,k,delta)
        k = k - grad * eps
    return(mp.renormalise2(k))

#mp.printimage([descente(mp.convol(k_3,exg.seuil_vect(mp.whitenoise(100,100,1))),k_2,0.000001,1)],["k_exp"])

def grad_e_sigma(Z,sigma,delta):
    M,N = np.shape(Z)
    k = noyau_gaussien_cached(sigma)
    k_prime = noyau_gaussien_prime_cached(sigma)
    result = 0
    R = 2*(ifft2(fft2(Z)/(fft2(k)+delta)) - (ifft2(fft2(Z)/(fft2(k)+delta)))**2)
    R = R * (2*ifft2(fft2(Z)/(fft2(k)+delta))- 1)
    R = R * (1/sigma**3 * ifft2(fft2(Z)*fft2(k_prime)/(fft2(k)**2+delta))- 2/sigma * ifft2(fft2(Z)/(fft2(k)+delta)))
    for i in range(M):
        for j in range(N):
            result += R[i,j] 
    return(np.real(result))

def descente_sigma(Z,sigma,delta,eps):
    for i in range(100):
        grad = grad_e_sigma(Z,sigma,delta)
        if grad < 0:
            sigma = sigma + eps
        else:
            sigma = max(sigma - eps, 0.5)
    return(sigma)

#print(descente_sigma(mp.convol(noyau_gaussien(5),exg.seuil_vect(mp.whitenoise(101,101,1))),5,0.001,0.1)) 

def plot_energie_sigma(Z, sigma_range, delta):
    energies = []
    for sigma in sigma_range:
        k = noyau_gaussien_cached(sigma)
        energy = energie(Z, k, delta)
        energies.append(energy)
    plt.plot(sigma_range, energies)
    plt.xlabel("Sigma")
    plt.ylabel("Energy")
    plt.title("Energy vs Sigma")
    plt.show()

#plot_energie_sigma(mp.convol(noyau_gaussien(5),exg.seuil_vectv2(mp.whitenoise(101,101,1))), np.linspace(0.1, 5.4, 100), 0.001)

def recherche_min (Z, sigma_range, delta):
    L = []
    for sigma in sigma_range:
        sigma_opt = descente_sigma(Z, sigma, delta, 0.1)
        k = noyau_gaussien_cached(sigma)
        energy = energie(Z, k, delta)
        L.append((energy, sigma_opt))
    L.sort(key=lambda x: x[0])
    print("Minimum energy:", L[0][0], "at sigma:", L[0][1])
    return(L)

print(recherche_min(mp.convol(noyau_gaussien(7),exg.seuil_vectv2(mp.whitenoise(101,101,1))), np.linspace(0.1, 10, 10), 0.001))
    
    
    
