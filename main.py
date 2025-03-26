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

def retrouver_k (u):  #u est une liste avec n réalisations de U
    n = len(u)
    N,M = u[0].shape
    gamma = np.zeros((N,M))
    for i in range(n):
        gamma += autocor(u[i])
    gamma = (1/(n*(N**2))) * gamma
    k = np.real(ifft2(np.sqrt(np.abs(fft2(gamma)))))
    return(k)

#printimage(retrouver_k([whitenoise(64,64,1) for i in range (5)]),"test_whitenoise")  #test pour k=1
w = [whitenoise(64,64,1) for i in range (10)]
#printimage(retrouver_k([convol_arr(k_arr,w[i]) for i in range(len(w))]),"test")   # on ne retrouve pas tout à fait k

def retrouver(u):  #u est une liste avec n réalisations de U
    n = len(u)
    N,M = u[0].shape
    gamma = np.zeros((N,M))
    for i in range(n):
        gamma += autocor(u[i])
    gamma = (1/(n*(N**2))) * gamma
    fft = np.sqrt(np.abs(fft2(gamma)))
    k0 = np.real(ifft2(fft))
    return k0, fftshift(fft), gamma, u[0]

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

            

#k1 = np.real(ifft2(np.abs(fft2(k))))
#fft_k1 = np.abs(fft2(k))
#printimage(fftshift(fft_k1), 'fftk1')
#plt.imshow(k)

#printimage(retrouver_k([whitenoise(64,64,1) for i in range (5)]),"test_whitenoise")  #test pour k=1
#w = [whitenoise(64,64,1) for i in range (500)]
#k0 = retrouver_k([convol_arr(k_arr,w[i]) for i in range(len(w))])
#u = [convol_arr(k_arr,w[i]) for i in range(len(w))]
#printimage(retrouver_k([convol_arr(k_arr,w[i]) for i in range(len(w))])[1],"test")   # on ne retrouve pas tout à fait k

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

        
wh = whitenoise(128,128,1)        
u = convol_arr(k_cer, wh)
printimage(u, 'convol_test')

#affiche(u)

