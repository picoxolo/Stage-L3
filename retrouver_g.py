import numpy as np
import main as mp
import exemples_k as ex
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from scipy import integrate
import pickle
from scipy.optimize import brentq
import bisect

with open("data.pkl", "rb") as f:
    data_loaded = pickle.load(f)
    
def proba_exp(U,n): #n:nombre de points
    sorted_array = np.sort(U.flatten())
    N = len(sorted_array)
    liste = [] #liste[0] = the upper value of the first (1/n+1) % ; liste[t]=the upper value of the first t*1/(n+1) %
    for i in range(n):
        liste.append(sorted_array[int(np.floor(i/(n+1)*N))])
    return (np.array(liste))

def f(x):
    return (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)

def phi(x):
    return(integrate.quad(f, -np.inf, x)[0])

def tab_phi(n):
    """
    n désigne le nombre de points sur lequel on estime phi
    on retourne T tq T[i] = phi^-1(i*99/n%)
    """
    points = [(i *  1/ (n+1))  for i in range(1, n + 1)]  # From 0% to 99%
    T = []
    for target in points:
    # Define the function for which we look for roots: phi(x) - target = 0
        inverse_x = brentq(lambda x: phi(x) - target, -10, 10)
        T.append(inverse_x)
    return T 

def plot_g_exp (tab_p,tab_phi,g,titre):
    n = len(tab_p)
    ordonnee = tab_p
    abscisse = tab_phi
    lin = np.linspace(abscisse[0],abscisse[-1],1000)
    plt.plot(abscisse,ordonnee, label = "g_exp")
    plt.plot(lin,g(lin), label = "g")
    plt.legend()
    plt.title(titre)
    plt.show()

def find_closest_index(l, x):
    pos = bisect.bisect_left(l, x)
    if pos == 0:
        return 0
    if pos == len(l):
        return len(l) - 1
    before = l[pos - 1]
    after = l[pos]
    if abs(after - x) < abs(x - before):
        return pos
    else:
        return pos - 1
    
def inverser_g (U,tab_p,tab_phi):
    n=len(tab_p)
    M,N = np.shape(U)
    for i in range(M):
        for j in range(N):
            r = find_closest_index(tab_p,U[i,j])
            U[i,j] = tab_phi[r]
    return(U)

def retrouve_tot(U, n, k0=np.array([[]]), g0=None, titre="", U0=None,a=0):
    tab_p=proba_exp(U,n)
    inv_phi = tab_phi(n)
    U_exp = inverser_g(U,tab_p,inv_phi)
    k_exp = mp.retrouver(U_exp)[0]
    ordonnee = tab_p
    abscisse = inv_phi
    lin = np.linspace(abscisse[0],abscisse[-1],1000)
    plt.plot(abscisse,ordonnee, label = "g_exp")
    if a!=0:
        plt.plot(lin,g0(lin), label = "g0")
        plt.title(titre)
    plt.legend()
    plt.show()  
    if a!=0: 
        mp.printimage([U_exp,U0, fftshift(k_exp),fftshift(k0)],
              ["exp","vrai","k_exp","k0"])
    else:
        mp.printimage([fftshift(k_exp)],
              ["k_exp"])
#%%
def F_prime(p):
    return 1/(2*np.pi*np.sqrt(1-p**2))*np.exp(-a**2/(1+p))

def F(p):
    return integrate.quad(F_prime, -1, p)[0] + phi(np.abs(a)) - phi(a)
    
#cf retour d'erreur si on sort de la plage
def F_inv(p):
    inverse_F = brentq(lambda x: F(x) - p, -30, 30)
    return inverse_F
    
def gamma_emp(u):
    M,N = u.shape
    aut = mn.autocor(u)
    gam = np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            gam[i,j] = F_inv(aut[i,j])
    return gam
    
