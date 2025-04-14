import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt

def whitenoise(N,M,sigma):
    noise = np.random.normal(0,sigma,(N,M))
    return(noise)

def printimage(img, title): #img et title sont des listes de meme longueur
    for i in range (len(img)):
        plt.figure()
        plt.title(title[i])
        plt.imshow(img[i], cmap="gray")
    plt.show()

def autocor(I):
    return np.real(ifft2(np.abs(fft2(I)**2)))

def convol(f,g):
    return np.real(ifft2(fft2(f)*fft2(g)))

def k0(C):
    return np.real(ifft2(np.sqrt(np.abs(fft2(C)))))

def retrouver(u):  
    n = len(u)
    N,M = u.shape
    gamma = autocor(u)/(N*M)
    fft = np.sqrt(np.abs(fft2(gamma)))
    k0 = np.real(ifft2(fft))
    return k0, fftshift(fft), gamma

def retrouver_K0(gamma):
    fft = np.sqrt(np.abs(fft2(gamma)))
    K_0 = np.real(ifft2(fft))
    return K_0

def affiche(u):
    plt.subplot(2,2,1)
    k0, fft, gamma, u = retrouver(u)
    plt.title('k0')
    plt.imshow(k0, cmap = 'gray')
    plt.subplot(2,2,2)
    plt.title('fft')
    plt.imshow(fft, cmap = 'gray')
    plt.subplot(2,2,3)
    plt.title('gamma')
    plt.imshow(gamma, cmap = 'gray')
    plt.subplot(2,2,4)
    plt.title('u')
    plt.imshow(u, cmap = 'gray')
    
def norme2(k):
    M,N = np.shape(k)
    norme = 0
    for i in range(M):
        for j in range(N):
            norme += k[i,j]**2
    return(norme**(1/2))

def norme1(k):
    return(np.sum(np.abs(k)))

def renormalise2(k):
    norme = norme2(k)
    return(k/norme)

def ecart_norme2 (k0,k):
    return(norme2(k0-k))

def ecart_norme1 (k0,k):
    return(norme1(k0-k))

def cut(k,seuil):
    M,N = np.shape(k)
    for i in range(M):
        for j in range(N):
            if k[i,j] < seuil : 
                k[i,j] = 0
    return(k)



def variance(exp,esp):
    M,N=np.shape(exp)
    v=0
    for i in range (M):
        for j in range(N):
            v += (exp[i,j] - esp[i,j])**2
    return v/(N*M)


def ecart_exp(gamma_ther, gamma_emp):
    M,N = gamma_emp.shape
    ecart_ther = np.sqrt(2/(M*N))*norme2(gamma_ther)
    ecart_emp = norme2(gamma_ther - gamma_emp)/np.sqrt(M*N)
    return np.abs(ecart_ther- ecart_emp)/np.abs(ecart_ther), ecart_ther, ecart_emp

#gamma_ther = autocor(k) #sigma = 1
