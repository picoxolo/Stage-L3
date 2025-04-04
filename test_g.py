import numpy as np
import main as mp
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from scipy import integrate

#on veut P(g(X)<=a)
def proba_exp (U,l,n): #l: demi-longueur de l'intervalle n: nombre de points impairs
    liste = np.array([0]*n, dtype=np.float64)  #liste qui découpe 
    M,N = np.shape(U)
    for i in range(M):
        for j in range(N):
            k = int(np.ceil((2*n/l)*U[i,j])) #on trouve le premier moment ou on est plus petit est on rajoute 1 dans la case
            if k + (n-1)//2 < n :
                liste[k + (n-1)//2 ] += 1
    c=0
    for i in range(n-1): #Puis on "propage" à tous les indices supérieurs
        c += liste[i]
        liste[i] += c - liste[i]
    return(liste/(N*M))

print(proba_exp (mp.whitenoise(1024,1024,1),2,20))

def f(x):
    return (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)


def tab_phi(c, n):
    """
    c tel que I = [-c,c], désigne la longueur du demi intervalle sur lequel on estime g
    n désigne le nombre de points, n impair
    """
    h = c/((n-1)/2)
    T = np.zeros(n)
    s = int((n-1)/2)
    T[s] = 0.5
    for k in range(s+1):
        T[s+k] = integrate.quad(f, -np.inf, k*h)[0]
        T[s-k] = integrate.quad(f, -np.inf, -k*h)[0]
    return T

def retrouver_g(tab_phi, tab_p,c,n):
    h = c/((n-1)/2)
    s = int((n-1)/2)
    N = len(tab_phi)
    T = np.zeros(n)
    i = 0
    j = 0
    while i < n:
        if tab_p[i] == 0.:
            while tab_p[i] == 0.:
                T[i] = -c
        while tab_p[i] > tab_phi[j]:
            j+=1
        if j <= s:
            T[i] = -(s-j)*h
        else:
            T[i] = j*h
        i+=1
    return T

#print(retrouver_g(tab_phi(5,1000),proba_exp(mp.whitenoise(2048,2048,1),4,1000),2,1000))

def plot_g (T,c,n):
    ordonnee = [(2*c/(n-1))*k for k in range ((-n+1)//2,((n+1)//2))]
    plt.plot(T,ordonnee)
    plt.show()
    
plot_g(retrouver_g(tab_phi(5,1000),proba_exp(mp.whitenoise(2048,2048,1),4,1000),2,1000),4,1000)
