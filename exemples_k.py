
import main as mp
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt

wh = mp.whitenoise(2048,2048,1)
'''
k_carre_petit = np.zeros((2048,2048))
for i in range(50):
    for j in range(50):
        k_carre_petit[1024-i,1024-j] = 1
        k_carre_petit[1024+i,1024+j] = 1
        k_carre_petit[1024+i,1024-j] = 1
        k_carre_petit[1024-i,1024+j] = 1
    
k_carre_petit_norme = mp.renormalise2(k_carre_petit)
k_carre_petit_conv_norme = mp.renormalise2(mp.convol (k_carre_petit_norme, k_carre_petit_norme))
u_conv_pca = mp.convol(k_carre_petit_conv_norme,wh)
k_exp_pca, fft_k_exp_pca, gamma_exp_pca = mp.retrouver(u_conv_pca)
k_exp_pca_2 = mp.renormalise2(k_exp_pca) #meilleur seuil 0.00085 ?
u_exp_pca_norma2 = mp.renormalise2(mp.convol(k_exp_pca_2,wh))
u_pca_norma2 = mp.renormalise2(u_conv_pca)
gamma_ther_pca = mp.autocor(k_carre_petit_conv_norme)
print("ecart gamma exp:", mp.ecart_exp(gamma_ther_pca, gamma_exp_pca))
print("ecart k :", mp.ecart_norme2(k_exp_pca_2,k_carre_petit_conv_norme))
print("ecart u:", mp.ecart_norme2(u_exp_pca_norma2,u_pca_norma2))
mp.printimage([
    fftshift(k_carre_petit_conv_norme),fftshift(k_exp_pca_2),fftshift(np.real(fft2(k_carre_petit_conv_norme))**2),
    fftshift(np.real(fft2(k_exp_pca_2))**2),u_exp_pca_norma2,u_pca_norma2
    ],
    ["petit_carré_convo_normé : k0",'k_exp','fft_petit_carré_convo_norme**2',"fft_kexp**2","u_exp","u"])
'''

'''
k_carre_grand = np.zeros((2048,2048))
for i in range(200):
    for j in range(200):
        k_carre_grand[1024-i,1024-j] = 1
        k_carre_grand[1024+i,1024+j] = 1
        k_carre_grand[1024+i,1024-j] = 1
        k_carre_grand[1024-i,1024+j] = 1
    
k_carre_grand_norme = mp.renormalise2(k_carre_grand)
k_carre_grand_conv_norme = mp.renormalise2(mp.convol (k_carre_grand_norme, k_carre_grand_norme))
u_conv_gca = mp.convol(k_carre_grand_conv_norme,wh)
k_exp_gca, fft_k_exp_gca, gamma_exp_gca = mp.retrouver(u_conv_gca)
k_exp_gca_2 = mp.renormalise2(k_exp_gca) #meilleur seuil 2.5 * 10^-5 ?
u_exp_gca_norma2 = mp.renormalise2(mp.convol(k_exp_gca_2,wh))
u_gca_norma2 = mp.renormalise2(u_conv_gca)
gamma_ther_gca = mp.autocor(k_carre_grand_conv_norme)
print("ecart gamma exp:", mp.ecart_exp(gamma_ther_gca, gamma_exp_gca))
print("ecart k :", mp.ecart_norme2(k_exp_gca_2,k_carre_grand_conv_norme))
print("ecart u :", mp.ecart_norme2(u_exp_gca_norma2,u_gca_norma2))
mp.printimage([
    fftshift(k_carre_grand_conv_norme),fftshift(k_exp_gca_2),fftshift(np.real(fft2(k_carre_grand_conv_norme))**2),
    fftshift(np.real(fft2(k_exp_gca_2))**2),u_exp_gca_norma2,u_gca_norma2
    ],
    ["grand_carré_convo_normé : k0",'k_exp','fft_grand_carré_convo_norme**2',"fft_kexp**2","u_exp","u"])
'''

'''
k_cer_tres_petit = np.zeros((2048,2048))
for i in range(2048):
    for j in range(2048):
        if (i-1024)**2 + (j-1024)**2 <= 100:
            k_cer_tres_petit[i,j] = 1

k_cer_tres_petit_conv = mp.convol (k_cer_tres_petit,k_cer_tres_petit)
k_cer_tres_petit_conv_norme = mp.renormalise2(k_cer_tres_petit_conv)
u_conv_tpce = mp.convol(k_cer_tres_petit_conv_norme,wh)
k_exp_tpce, fft_k_exp_tpce, gamma_exp_tpce = mp.retrouver(u_conv_tpce)
gamma_tpce = mp.autocor(k_cer_tres_petit_conv_norme)
k_exp_tpce_2 = mp.renormalise2(k_exp_tpce) # meilleur seuil 7.25 * 10^-4
u_exp_tpce_norma2 = mp.renormalise2(mp.convol(k_exp_tpce_2,wh))
u_tpce_norma2 = mp.renormalise2(u_conv_tpce)
gamma_ther_tpce = mp.autocor(k_cer_tres_petit_conv_norme)
print("ecart gamma exp:", mp.ecart_exp(gamma_ther_tpce, gamma_exp_tpce))
print("ecart k :", mp.ecart_norme2(k_exp_tpce_2,k_cer_tres_petit_conv_norme))
print("ecart u :", mp.ecart_norme2(u_exp_tpce_norma2,u_tpce_norma2))
print("norme2 de gamma_exp :",mp.norme2(gamma_exp_tpce),"  norme2 de gamma :",mp.norme2(gamma_tpce))
print("ecart type sur gamma(x) =", (2*(mp.norme2(gamma_tpce)**2)/(2048**2))**(1/2))
print("ecart type exp gamma(x) =", mp.variance(gamma_exp_tpce,gamma_tpce)**(1/2))
#mp.printimage([fftshift(gamma_tpce),fftshift(gamma_exp_tpce)],["gamma","gamma_exp"])
mp.printimage([
    fftshift(k_cer_tres_petit_conv_norme),fftshift(k_exp_tpce_2),fftshift(np.real(fft2(k_cer_tres_petit_conv_norme))**2),
    fftshift(np.real(fft2(k_exp_tpce_2))**2),u_exp_tpce_norma2,u_tpce_norma2
    ],
    ["très_petit_cercle_convo_normé : k0",'k_exp','fft_très_petit_cercle_convo_norme**2',"fft_kexp**2","u_exp","u"])
'''

'''
k_cer_petit = np.zeros((2048,2048))
for i in range(2048):
    for j in range(2048):
        if (i-1024)**2 + (j-1024)**2 <= 1000:
            k_cer_petit[i,j] = 1
 
k_cer_petit_conv = mp.convol (k_cer_petit,k_cer_petit)
k_cer_petit_conv_norme = mp.renormalise2(k_cer_petit_conv)
u_conv_pce = mp.convol(k_cer_petit_conv_norme,wh)
k_exp_pce, fft_k_exp_pce, gamma_exp_pce = mp.retrouver(u_conv_pce)
gamma_pce = mp.autocor(k_cer_petit_conv_norme)
k_exp_pce_2 = mp.renormalise2(k_exp_pce) # meilleur seuil 8,5 * 10^-4 ?
u_exp_pce_norma2 = mp.renormalise2(mp.convol(k_exp_pce_2,wh))
u_pce_norma2 = mp.renormalise2(u_conv_pce)
gamma_ther_pce = mp.autocor(k_cer_petit_conv_norme)
print("ecart gamma exp:", mp.ecart_exp(gamma_ther_pce, gamma_exp_pce))
print("ecart k :", mp.ecart_norme2(k_exp_pce_2,k_cer_petit_conv_norme))
print("ecart u :", mp.ecart_norme2(u_exp_pce_norma2,u_pce_norma2))
print("norme2 gamma_exp =",mp.norme2(gamma_exp_pce),"  norme2 gamma =",mp.norme2(gamma_pce))
print("ecart type sur gamma(x) =", (2*(mp.norme2(gamma_pce)**2)/(2048**2))**(1/2))
print("ecart type exp gamma(x) =", mp.variance(gamma_exp_pce,gamma_pce)**(1/2))
mp.printimage([fftshift(gamma_pce),fftshift(gamma_exp_pce)],["gamma","gamma_exp"])
mp.printimage([
    fftshift(k_cer_petit_conv_norme),fftshift(k_exp_pce_2),fftshift(np.real(fft2(k_cer_petit_conv_norme))**2),
    fftshift(np.real(fft2(k_exp_pce_2))**2),u_exp_pce_norma2,u_pce_norma2
    ],
    ["petit_cercle_convo_normé : k0",'k_exp','fft_petit_cercle_convo_norme**2',"fft_kexp**2","u_exp","u"])
'''

'''
k_cer_grand = np.zeros((2048,2048))
for i in range(2048):
    for j in range(2048):
        if (i-1024)**2 + (j-1024)**2 <= 160000:
            k_cer_grand[i,j] = 1
        
k_cer_grand_conv = mp.convol (k_cer_grand,k_cer_grand)
k_cer_grand_conv_norme = mp.renormalise2(k_cer_grand_conv)
u_conv_gce = mp.convol(k_cer_grand_conv_norme,wh)
k_exp_gce, fft_k_exp_gce, gamma_exp_gce = mp.retrouver(u_conv_gce)
gamma_gce = mp.autocor(k_cer_grand_conv_norme)
k_exp_gce_2 = mp.renormalise2(k_exp_gce) # meilleur seuil 1,25 * 10^-4 ?
u_exp_gce_norma2 = mp.renormalise2(mp.convol(k_exp_gce_2,wh))
u_gce_norma2 = mp.renormalise2(u_conv_gce)
gamma_ther_gce = mp.autocor(k_cer_grand_conv_norme)
print("ecart gamma exp:", mp.ecart_exp(gamma_ther_gce, gamma_exp_gce))
print("ecart k :", mp.ecart_norme2(k_exp_gce_2,k_cer_grand_conv_norme))
print("ecart u :", mp.ecart_norme2(u_exp_gce_norma2,u_gce_norma2))
print("norme2 gamma_exp =",mp.norme2(gamma_exp_gce),"  norme2 gamma =",mp.norme2(gamma_gce))
print("ecart type sur gamma(x) =", (2*(mp.norme2(gamma_gce)**2)/(2048**2))**(1/2))
print("ecart type exp gamma(x) =", mp.variance(gamma_exp_gce,gamma_gce)**(1/2))
mp.printimage([fftshift(gamma_gce),fftshift(gamma_exp_gce)],["gamma","gamma_exp"])
mp.printimage([
    fftshift(k_cer_grand_conv_norme),fftshift(k_exp_gce_2),fftshift(np.real(fft2(k_cer_grand_conv_norme))**2),
    fftshift(np.real(fft2(k_exp_gce_2))**2),u_exp_gce_norma2,u_gce_norma2
    ],
    ["grand_cercle_convo_normé : k0",'k_exp','fft_grand_cercle_convo_norme**2',"fft_kexp**2","u_exp","u"])
'''

'''
k_dirac = np.zeros((2048,2048))
k_dirac[1024,1024] = 1
k_dirac_conv = mp.convol (k_dirac,k_dirac)
k_dirac_conv_norme = mp.renormalise2(k_dirac_conv)
u_conv_dirac = mp.convol(k_dirac_conv_norme,wh)
k_exp_dirac, fft_k_exp_dirac, gamma_exp_dirac = mp.retrouver(u_conv_dirac)
gamma_dirac = mp.autocor(k_dirac_conv_norme)
k_exp_dirac_2 = mp.renormalise2(k_exp_dirac)
u_exp_dirac_norma2 = mp.renormalise2(mp.convol(k_exp_dirac_2,wh))
u_dirac_norma2 = mp.renormalise2(u_conv_dirac)
gamma_ther_dirac = mp.autocor(k_dirac_conv_norme)
print("ecart gamma exp:", mp.ecart_exp(gamma_ther_dirac, gamma_exp_dirac))
print("ecart k :", mp.ecart_norme2(k_exp_dirac_2,k_dirac_conv_norme))
print("ecart u :", mp.ecart_norme2(u_exp_dirac_norma2,u_dirac_norma2))
print("norme2 gamma_exp =",mp.norme2(gamma_exp_dirac),"  norme2 gamma =",mp.norme2(gamma_dirac))
print("ecart type sur gamma(x) =", (2*(mp.norme2(gamma_dirac)**2)/(2048**2))**(1/2))
print("ecart type exp gamma(x) =", mp.variance(gamma_exp_dirac,gamma_dirac)**(1/2))
mp.printimage([fftshift(gamma_dirac),fftshift(gamma_exp_dirac)],["gamma","gamma_exp"])
mp.printimage([
    fftshift(k_dirac_conv_norme),fftshift(k_exp_dirac_2),fftshift(np.real(fft2(k_dirac_conv_norme))**2),
    fftshift(np.real(fft2(k_exp_dirac_2))**2),u_exp_dirac_norma2,u_dirac_norma2
    ],
    ["dirac_convo_normé : k0",'k_exp','fft_dirac_convo_norme**2',"fft_kexp**2","u_exp","u"])
'''
'''
k_cer_petit = np.zeros((2048,2048))
for i in range(2048):
    for j in range(2048):
        if (i-1024)**2 + (j-1024)**2 <= 1000:
            k_cer_petit[i,j] = 1
k_cer_petit_conv = mp.convol (k_cer_petit,k_cer_petit)
k_cer_petit_conv_norme = mp.renormalise2(k_cer_petit_conv)
u_conv_pce = mp.convol(k_cer_petit_conv_norme,wh)
k_exp_pce, fft_k_exp_pce, gamma_exp_pce = mp.retrouver(u_conv_pce)
gamma_pce = mp.autocor(k_cer_petit_conv_norme)
k_exp_pce_2 = mp.renormalise2(k_exp_pce) # meilleur seuil 8,5 * 10^-4 ?
u_exp_pce_norma2 = mp.renormalise2(mp.convol(k_exp_pce_2,wh))
u_pce_norma2 = mp.renormalise2(u_conv_pce)
print("ecart k :", mp.ecart_norme2(k_exp_pce_2,k_cer_petit_conv_norme))
print("ecart u :", mp.ecart_norme2(u_exp_pce_norma2,u_pce_norma2))
print("norme2 gamma_exp =",mp.norme2(gamma_exp_pce),"  norme2 gamma =",mp.norme2(gamma_pce))
print("ecart type sur gamma(x) =", (2*(mp.norme2(gamma_pce)**2)/(2048**2))**(1/2))
print("ecart type exp gamma(x) =", mp.variance(gamma_exp_pce,gamma_pce)**(1/2))
mp.printimage([fftshift(gamma_pce),fftshift(gamma_exp_pce)],["gamma","gamma_exp"])
mp.printimage([
    fftshift(k_cer_petit_conv_norme),fftshift(k_exp_pce_2),fftshift(np.real(fft2(k_cer_petit_conv_norme))**2),
    fftshift(np.real(fft2(k_exp_pce_2))**2),u_exp_pce_norma2,u_pce_norma2
    ],
    ["petit_cercle_convo_normé : k0",'k_exp','fft_petit_cercle_convo_norme**2',"fft_kexp**2","u_exp","u"])
mp.plot_norme(k_cer_petit_conv_norme, k_exp_pce_2) 
'''
