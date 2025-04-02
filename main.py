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
    
def norme2(k):
    M,N = np.shape(k)
    norme = 0
    for i in range(M):
        for j in range(N):
            norme += k[i,j]**2
    return(norme**(1/2))

def norme1(k):
    return(np.sum(np.abs(k)))

def renormalise(k):
    norme = norme2(k)
    return(k/norme)

wh = whitenoise(2048,2048,1)

#%%
k_carre_petit = np.zeros((2048,2048))
for i in range(50):
    for j in range(50):
        k_carre_petit[1024-i,1024-j] = 1
        k_carre_petit[1024+i,1024+j] = 1
        k_carre_petit[1024+i,1024-j] = 1
        k_carre_petit[1024-i,1024+j] = 1
    
k_carre_petit_norme = renormalise(k_carre_petit)
k_carre_petit_conv_norme = renormalise(convol (k_carre_petit_norme, k_carre_petit_norme))


k_carre_grand = np.zeros((2048,2048))
for i in range(200):
    for j in range(200):
        k_carre_grand[1024-i,1024-j] = 1
        k_carre_grand[1024+i,1024+j] = 1
        k_carre_grand[1024+i,1024-j] = 1
        k_carre_grand[1024-i,1024+j] = 1
    
k_carre_grand_norme = renormalise(k_carre_grand)
k_carre_grand_conv_norme = renormalise(convol (k_carre_grand_norme, k_carre_grand_norme))

k_cer_tres_petit = np.zeros((2048,2048))
for i in range(2048):
    for j in range(2048):
        if (i-1024)**2 + (j-1024)**2 <= 100:
            k_cer_tres_petit[i,j] = 1

k_cer_tres_petit_conv = convol (k_cer_tres_petit,k_cer_tres_petit)
k_cer_tres_petit_conv_norme = renormalise(k_cer_tres_petit_conv)


k_cer_petit = np.zeros((2048,2048))
for i in range(2048):
    for j in range(2048):
        if (i-1024)**2 + (j-1024)**2 <= 1000:
            k_cer_petit[i,j] = 1

            
k_cer_petit_conv = convol (k_cer_petit,k_cer_petit)
k_cer_petit_conv_norme = renormalise(k_cer_petit_conv)

k_cer_grand = np.zeros((2048,2048))
for i in range(2048):
    for j in range(2048):
        if (i-1024)**2 + (j-1024)**2 <= 160000:
            k_cer_grand[i,j] = 1
            
k_cer_grand_conv = convol (k_cer_grand,k_cer_grand)
k_cer_grand_conv_norme = renormalise(k_cer_grand_conv)
#%%
              
u_con = convol(k_cer_tres_petit_conv_norme,wh)
kexp, fft, gamma, u = retrouver(u_con)

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

kexp2 = renormalise(cut(kexp,0.001))  #comment choisir le seuil ???
u_exp_norma = renormalise(convol(kexp2,wh))
u_con_norma = renormalise(u_con)
print(ecart_norme2(k_cer_tres_petit_conv_norme,kexp2))
print(ecart_norme2(u_con_norma,u_exp_norma))
printimage([fftshift(k_cer_tres_petit_conv_norme),fftshift(kexp2),fftshift(np.real(fft2(k_cer_tres_petit_conv_norme))**2),fftshift(np.real(fft2(kexp2))**2),u_exp_norma,u_con_norma],
          ["trespetitcercleconvnorme",'kexp','ffttrespetitcercleconvnorme**2',"fftkexp**2","uexp","ucon"])







def variance(fftk0,fftkexp):
    N,M=np.shape(k0)
    v=0
    for i in range (N):
        for j in range(M):
            if fftk0[i,j]<10**(-2):
                v+=1
            else : 
                v+= (fftk0[i,j]**2-fftkexp[i,j]**2)/(fftk0[i,j]**2)
    return(v/(N*M))
    
#print(variance(fft_k0,fft))
#printimage(u, 'convol_test')


def ecart_exp(gamma_ther, gamma_emp):
    M,N = gamma_emp.shape
    ecart_ther = np.sqrt(2/(M*N))*norme_2(gamma_ther)
    ecart_emp = norme_2(gamma_ther - gamma_emp)/np.sqrt(M*N)
    return np.abs(ecart_ther- ecart_emp)/np.abs(ecart_ther), ecart_ther, ecart_emp

#gamma_ther = autocor(k) #sigma = 1
