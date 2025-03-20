import numpy as np
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