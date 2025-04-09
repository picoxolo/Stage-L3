import numpy as np
import main as mp
import exemples_k as ex
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from scipy import integrate
import pickle

with open("data.pkl", "rb") as f:
    data_loaded = pickle.load(f)

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
    return(liste/(N*M)) #On obtient une liste avec n point tq l[k]=P(g(X)<=(k-(n-1)/2)*l*/(n-1))=F(g^-1(a))

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
    return T #on obtient un array avec les valeurs de F (phi) au points k*h

def retrouver_g(tab_phi, tab_p,l_phi):
    N = len(tab_phi)
    h = 2*l_phi/(N-1)
    s = int((N-1)/2)
    n=len(tab_p)
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
    return T #on obient T[k]=g^-1(k-(n-1)/2)*h)

def plot_g_exp (T,l,g,titre):
    n = len(T)
    ordonnee = np.array([(2*l/(n-1))*k for k in range ((-n+1)//2,((n+1)//2),1)])
    lin = np.linspace(T[0],T[-1],1000)
    plt.plot(T,ordonnee, label = "g_exp")
    plt.plot(lin,g(lin), label = "g")
    plt.legend()
    plt.title(titre)
    plt.show()

def inverser_g (T,U,l):
    n=len(T)
    M,N = np.shape(U)
    for i in range(M):
        for j in range(N):
            k=int(np.floor((n-1)*U[i,j]/(2*l)))
            U[i,j] = T[(k+(n-1)//2)]
    return(U)


def retrouve_tot(U, k0=0, g0=0, titre="", U0=0):
    T=retrouver_g(data_loaded["tab_phi"],proba_exp(U,5,10001),5)
    U_exp = inverser_g(T,U,5)
    k_exp = mp.retrouver(U_exp)[0]
    n = len(T)
    ordonnee = np.array([(2*5/(n-1))*k for k in range ((-n+1)//2,((n+1)//2),1)])
    lin = np.linspace(T[0],T[-1],1000)
    plt.plot(T,ordonnee, label = "g_exp")
    if k0!=0:
        plt.plot(lin,g0(lin), label = "g0")
        plt.title(titre)
    plt.legend()
    plt.show()  
    if k0 !=0: 
        mp.printimage([U_exp,U0, fftshift(k_exp),fftshift(k0)],
              ["exp","vrai","k_exp","k0"])
    else:
        mp.printimage([fftshift(k_exp)],
              ["k_exp"])