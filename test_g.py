import numpy as np
import main as mp
import exemples as ex
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from scipy import integrate

#on veut P(g(X)<=a)
def proba_exp (U,l,n): #l: demi-longueur de l'intervalle n: nombre de points impairs
    liste = np.array([0]*n, dtype=np.float64)  #liste qui découpe [-l,l] en n points
    M,N = np.shape(U)
    for i in range(M):
        for j in range(N):
            k = int(np.ceil(((n-1)/(2*l))*U[i,j])) #on trouve le premier moment ou on est plus petit est on rajoute 1 dans la case
            if (k + (n-1)//2 <n) and (k + (n-1)//2 >= 0): 
                liste[k + (n-1)//2 ] += 1
    c=0
    for i in range(n): #Puis on "propage" à tous les indices supérieurs
        c += liste[i]
        liste[i] += c - liste[i]
    return(liste/(N*M))

def f(x):
    return (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)

def tab_phi(l, n):
    """
    l tel que I = [-l,l], désigne la longueur du demi intervalle sur lequel on estime g
    n désigne le nombre de points, n impair
    """
    h = 2*l/(n-1)
    T = np.zeros(n)
    s = int((n-1)/2)
    T[s] = 0.5
    for k in range(1,s+1):
        T[s+k] = integrate.quad(f, -np.inf, k*h)[0]
        T[s-k] = integrate.quad(f, -np.inf, -k*h)[0]
    return T

def retrouver_g(tab_phi, tab_p,l_phi,n):
    N = len(tab_phi)
    h = 2*l_phi/(N-1)
    s = int((N-1)/2)
    T = np.zeros(n)
    i = 0
    j = 0
    while tab_p[i] == 0.:
        T[i] = -l_phi
        i +=1
    while i < n:
        while (tab_p[i] > tab_phi[j]) and (j<N-1):
            j+=1
        T[i] = (-s+j)*h
        i+=1
    return T

def plot_g_exp (T,l,n,g,titre):
    ordonnee = np.array([(2*l/(n-1))*k for k in range ((-n+1)//2,((n+1)//2),1)])
    l = np.linspace(T[0],T[-1],1000)
    plt.plot(T,ordonnee, label = "g_exp")
    plt.plot(l,g(l), label = "g")
    plt.legend()
    plt.title(titre)
    plt.show()
