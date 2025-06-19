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
    liste = [] #liste[0] = the upper value of the first (100/n+1) % ; liste[t]=the upper value of the first t*100/(n+1) %
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
    points = [(i *  1/ (n+1))  for i in range(1, n + 1)]  # From 1% to 99%
    T = []
    for target in points:
    # Define the function for which we look for roots: phi(x) - target = 0
        inverse_x = brentq(lambda x: phi(x) - target, -10, 10)
        T.append(inverse_x)
    return T 

tab_phi(10000)

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
    
def inverser_g(U,tab_p,tab_phi):
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
    
    
def F_prime(a,p):
    a,p = np.float64(a),np.float64(p)
    return 1/(2*np.pi*np.sqrt(1-p**2))*np.exp(-a**2/(1+p))

def F(a,p):
    def g(p):
        return F_prime(a,p)
    if p==-1:
        return(phi(np.abs(a)) - phi(a))
    else:
        return integrate.quad(g, -1, p)[0] + phi(np.abs(a)) - phi(a)


#cf retour d'erreur si on sort de la plage
def F_inv(a,p):
    def g(x):
        return F(a,x)
    inverse_F = brentq(lambda x: g(x) - p, -1, 1)
    return inverse_F

def F_inv_main(a,p,n):
    f = [F(a,-1 + i/n) for i in range(2*n+1)]
    index = find_closest_index(f,p)
    return(-1+index/n)

def plot_F_F_inv(n):
    a = [0.0, 0.25, 0.50, 0.75, 1]
    lis_f = []
    abscisse = np.array([(-1+i/n) for i in range(2*n+1)])
    for i in a:
        def g(x):
            return F(i,x)
        g_vect = np.vectorize(g)
        plt.plot(g_vect(abscisse),abscisse)
    plt.title(F_inv)
    plt.show()
    
def plot_H_H_invv(n):
    a = [0.0, 0.25, 0.50, 0.75, 1]
    lis_h = []
    abscisse = np.array([(-1+i/n) for i in range(2*n+1)])
    plt.figure(figsize=(10, 6))  # Définir la taille de la figure
    for i in a:
        def g(x):
            return F(i,x)
        g_vect = np.vectorize(g)
        plt.plot(g_vect(abscisse), abscisse, label=f'a = {i}')  # Ajouter un label pour chaque courbe
    plt.title(r"Graphes de $H_{a}^{-1}$")  # Mettre le titre entre guillemets
    plt.xlabel("x")  # Ajouter un label pour l'axe x
    plt.ylabel(r"$H^{-1}(x)$")  # Ajouter un label pour l'axe y
    plt.legend(loc="lower right", prop={"size" : 15})  # Afficher la légende
    plt.show()

#plot_H_H_invv(1000)
    
    
def gamma_emp(a,v,n): #n precision pour f^-1
    M,N = v.shape
    k,l = 0,0
    c1,c2 = v[0][0], v[0][0]
    while ((k < M) and c1==c2):
        while (l < N and c1==c2):
            if v[k,l] < c1:
                c1 = v[k,l]
            if v[k,l] > c1:
                c2 = v[k,l]
            l +=1
        k+=1
        l=0
    if c1==c2:
        raise Exception("Constant image")
    else:
        v = (v-c1)/(c2-c1)  #on obtient une image avec des 0 et des 1
    aut = mp.autocor(v)/(M*N)
    gam = np.zeros((M,N))
    f = [F(a,-1 + i/n) for i in range(2*n+1)]
    for i in range(M):
        for j in range(N):
            index = find_closest_index(f,aut[i,j])
            gam[i,j] = -1+index/n
    return gam


def retrouve_seuil(v,n):
    t_phi, t_p = tab_phi(n), proba_exp(v,n) #t_phi = abscisses, t_p = ordonnées
    index = np.searchsorted(t_p,t_p[-1]) #dichotomie, O(log n)
    return t_phi[index] #a

def retrouve_tot_seuil(v,n,k0=np.array([[]]), g0=None, titre="", t=0):#n precision sur tab_phi
    #k et g = seuil
    M,N = np.shape(v)
    t_phi, t_p = tab_phi(n), proba_exp(v,n) #t_phi = abscisses, t_p = ordonnées
    index = np.searchsorted(t_p,t_p[-1]) #dichotomie, O(log n)  
    a = t_phi[index]
    print(a)
    k_exp = mp.renormalise2(mp.retrouver_K0(gamma_emp(a,v,n)))
    ordonnee = t_p
    abscisse = t_phi
    lin = np.linspace(abscisse[0],abscisse[-1],1000)
    plt.plot(abscisse,ordonnee, label = "g_exp")
    if t!=0:
        plt.plot(lin,g0(lin), label = "g0")
        plt.title(titre)
    plt.legend()
    plt.show()  
    if t!=0: 
        mp.printimage([fftshift(k_exp),fftshift(k0)],
              ["k_exp","k0"])
    else:
        mp.printimage([fftshift(k_exp)],
              ["k_exp"])

def retrouve_Relu(Y,T):
    #T = rtg.phi_b(n)
    M,N = np.shape(Y)
    n = np.shape(T)[0]
    tab = np.sort(Y.flatten())
    i = bisect.bisect_right(tab, 0)
    if i == len(tab):
        i = i - 1
    P = i/(M*N)
    a = T[int(np.floor((n+1)*P))]
    l = M*N - i
    c = tab[i+l//2]
    p = (i/(M*N)+1)/2
    b = (T[int(np.floor((n+1)*p))]-a)*c
    return a,b

    
