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
k_exp_pca, fft_k_exp_pca, gamma_pca = mp.retrouver(u_conv_pca)
k_exp_pca_2 = mp.renormalise2(mp.cut(k_exp_pca,0.00085)) #meilleur seuil 0.00085 ?
u_exp_pca_norma2 = mp.renormalise2(mp.convol(k_exp_pca_2,wh))
u_pca_norma2 = mp.renormalise2(u_conv_pca)
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
k_exp_gca, fft_k_exp_gca, gamma_gca = mp.retrouver(u_conv_gca)
k_exp_gca_2 = mp.renormalise2(mp.cut(k_exp_gca,0.000025)) #meilleur seuil 2.5 * 10^-5 ?
u_exp_gca_norma2 = mp.renormalise2(mp.convol(k_exp_gca_2,wh))
u_gca_norma2 = mp.renormalise2(u_conv_gca)
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
k_exp_tpce, fft_k_exp_tpce, gamma_tpce = mp.retrouver(u_conv_tpce)
k_exp_tpce_2 = mp.renormalise2(mp.cut(k_exp_tpce,0.000725)) # meilleur seuil 7.25 * 10^-4
u_exp_tpce_norma2 = mp.renormalise2(mp.convol(k_exp_tpce_2,wh))
u_tpce_norma2 = mp.renormalise2(u_conv_tpce)
print("ecart k :", mp.ecart_norme2(k_exp_tpce_2,k_cer_tres_petit_conv_norme))
print("ecart u :", mp.ecart_norme2(u_exp_tpce_norma2,u_tpce_norma2))
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
k_exp_pce, fft_k_exp_pce, gamma_pce = mp.retrouver(u_conv_pce)
k_exp_pce_2 = mp.renormalise2(mp.cut(k_exp_pce,0.00085)) # meilleur seuil 8,5 * 10^-4 ?
u_exp_pce_norma2 = mp.renormalise2(mp.convol(k_exp_pce_2,wh))
u_pce_norma2 = mp.renormalise2(u_conv_pce)
print("ecart k :", mp.ecart_norme2(k_exp_pce_2,k_cer_petit_conv_norme))
print("ecart u :", mp.ecart_norme2(u_exp_pce_norma2,u_pce_norma2))
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
k_exp_gce, fft_k_exp_gce, gamma_gce = mp.retrouver(u_conv_gce)
k_exp_gce_2 = mp.renormalise2(mp.cut(k_exp_gce,0.000125)) # meilleur seuil 1,25 * 10^-4 ?
u_exp_gce_norma2 = mp.renormalise2(mp.convol(k_exp_gce_2,wh))
u_gce_norma2 = mp.renormalise2(u_conv_gce)
print("ecart k :", mp.ecart_norme2(k_exp_gce_2,k_cer_grand_conv_norme))
print("ecart u :", mp.ecart_norme2(u_exp_gce_norma2,u_gce_norma2))
mp.printimage([
    fftshift(k_cer_grand_conv_norme),fftshift(k_exp_gce_2),fftshift(np.real(fft2(k_cer_grand_conv_norme))**2),
    fftshift(np.real(fft2(k_exp_gce_2))**2),u_exp_gce_norma2,u_gce_norma2
    ],
    ["grand_cercle_convo_normé : k0",'k_exp','fft_grand_cercle_convo_norme**2',"fft_kexp**2","u_exp","u"])
'''