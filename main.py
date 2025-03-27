import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt

def whitenoise(N,M,sigma):
    noise = np.random.normal(0,sigma,(N,M))
    return(noise)

def printimage(img, title):
    plt.title(title)
    plt.imshow(img, cmap="gray")
    plt.show()

#printimage(whitenoise(64,64,1),"whitenoise")

k_dict ={(0,0):1, (1,0):1}

def convol_dict(k,W):
    N,M = np.shape(W)
    U = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            for (m,n) in k.keys():
                U[i][j] += k[(m,n)]*W[i-m][j-n]
    return U

k_arr = np.array([[1,1], [1,1]])

def convol_arr(k,W):
    N,M = np.shape(W)
    V,X = np.shape(k)
    U = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            for m in range(V):
                for n in range(X):
                    U[i][j] += k[m,n]*W[i-m][j-n]
    return U



def auto_cor(I): #U est réelle
    N,M = I.shape
    A = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            for m in range(N):
                for n in range(M):
                    A[i][j] += I[(i+m)%N][(j+n)%M]*I[m][n]
    return A

def autocor(I):
    return np.real(ifft2(np.abs(fft2(I)**2)))

def convol(f,g):
    return np.real(ifft2(fft2(f)*fft2(g)))

def k0(C):
    return np.real(ifft2(np.sqrt(np.abs(fft2(C)))))

#printimage(autocor(convol_dict(k_dict,whitenoise(64,64,1))),"autocorrelationwhitenoise")

def retrouver_k (u):  #u est une liste avec n réalisations de U
    n = len(u)
    N,M = u[0].shape
    gamma = np.zeros((N,M))
    for i in range(n):
        gamma += autocor(u[i])
    gamma = (1/(n*(N**2))) * gamma
    k = np.real(ifft2(np.sqrt(np.abs(fft2(gamma)))))
    return(k)

def retrouver(u):  
    n = len(u)
    N,M = u.shape
    gamma = autocor(u)/(N*M)
    fft = np.sqrt(np.abs(fft2(gamma)))
    k0 = np.real(ifft2(fft))
    return k0, fftshift(fft), gamma, u

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

k_t = np.zeros((8,8))
for i in range(8):
    for j in range(8):
        k_t[i,j] = 1/64
        
k_cer = np.zeros((64,64))
for i in range(64):
    for j in range(64):
        if (i-64)**2 + (j-64)**2 <= 64:
            k_cer[i,j] = 1
#printimage(k_cer, 'cercle')

k = np.zeros((2048,2048))
k[0,1], k[0,0], k[1,0], k[1,1] = 1,1,1,1        

k1 = np.real(ifft2(np.abs(fft2(k))))
fft_k1 = np.abs(fft2(k))
printimage(np.abs(fftshift(fft_k1)), 'fftk1')
        
wh = whitenoise(2048,2048,1)        
u_con = convol(k,wh)

k0, fft, gamma, u = retrouver(u_con)
printimage(fft, 'abs_fftshift_k0')  

printimage(u, 'convol_test')

def norme_2(u):
    M,N = u.shape
    S = 0
    for i in range(M):
        for j in range(N):
            S += np.abs(u[i,j])**2
    return np.sqrt(S)

def seuil(gamma_ther, gamma_emp):
    M,N = gamma_emp.shape
    ecart_ther = np.sqrt(2/(M*N))*norme_2(gamma_ther)
    ecart_emp = norme_2(gamma_ther - gamma_emp)/np.sqrt(M*N)
    return np.abs(ecart_ther- ecart_emp)/np.abs(ecart_ther), ecart_ther, ecart_emp

gamma_ther = autocor(k) #sigma = 1
