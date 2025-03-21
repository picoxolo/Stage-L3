import numpy as np
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt

def whitenoise(N,M,sigma):
    noise = np.random.normal(0,sigma,(N,M))
    return(noise)

def printimage(img, title):
    plt.title(title)
    plt.imshow(img, cmap="gray")
    plt.show()

#printimage(whitenoise(64,64,1),"whitenoise")

def convolution(k,W):
    N = len(W)
    M = len(W[0])
    U = np.array([[0.]*M for i in range(N)])
    for i in range (N):
        for j in range (M):
            for l in range (N):
                for m in range(M):
                    U[i,j] += (k[l,m]*W[(i-l)%(N),(j-m)%(M)])
    return(U)

k = np.array([[0.]*64 for i in range(64)])
k[0,0] = 1
k[0,1] = 1

#printimage(convolution(k,whitenoise(64,64,1)),"convolution")

def autocorrelation(I,m,l):
    N,M = I.shape
    A=0
    for i in range(N):
        for j in range(M):
            A += I[(i+m)%N,(j+l)%M]*I[i,j]
    return(A)

#print(autocorrelation(whitenoise(64,64,1),1,0))
    
def matriceautocorrelation(I):
    N,M = I.shape
    A = np.array([[0.]*M for i in range(N)])
    for i in range(N):
        for j in range(M):
            A[i,j] = autocorrelation(I,i,j)
    return(A)

#printimage(matriceautocorrelation(convolution(k,whitenoise(64,64,1))),"autocorrelationwhitenoise")

#%%
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

 def k0(C):
    return np.real(ifft2(np.sqrt(np.abs(fft2(C)))))

#printimage(autocor(convol_dict(k_dict,whitenoise(64,64,1))),"autocorrelationwhitenoise")
