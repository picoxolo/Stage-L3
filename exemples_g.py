import numpy as np
import main as mp
import exemples_k as ex
import retrouver_g as rtg
import matplotlib.pyplot as plt
import pickle
from numpy.fft import fft2, ifft2, fftshift

with open("data.pkl", "rb") as f:
    data_loaded = pickle.load(f)

def id(x):
    return(x)
"""
rtg.plot_g_exp(rtg.retrouver_g(data_loaded["tab_phi"],rtg.proba_exp(id(data_loaded["u_conv_tpce"]),5,1001),5),
               5,id,"g(x) = x")

rtg.plot_g_exp(rtg.retrouver_g(data_loaded["tab_phi"],rtg.proba_exp(id(data_loaded["u_conv_pce"]),5,1001),5),
               5,id,"g(x) = x")

rtg.plot_g_exp(rtg.retrouver_g(data_loaded["tab_phi"],rtg.proba_exp(id(data_loaded["u_conv_gce"]),5,1001),5),
               5,id,"g(x) = x")
"""

def id2(x):
    return(2*x)
'''
rtg.plot_g_exp(rtg.retrouver_g(data_loaded["tab_phi"],rtg.proba_exp(id2(data_loaded["u_conv_tpce"]),5,1001),5),
              5,id2,"g(x) = 2x")

rtg.plot_g_exp(rtg.retrouver_g(data_loaded["tab_phi"],rtg.proba_exp(id2(data_loaded["u_conv_pce"]),5,1001),5),
              5,id2,"g(x) = 2x")

rtg.plot_g_exp(rtg.retrouver_g(data_loaded["tab_phi"],rtg.proba_exp(id2(data_loaded["u_conv_gce"]),5,1001),5),
              5,id2,"g(x) = 2x")
'''

def x3(x):
    return(x**3)
'''
rtg.plot_g_exp(rtg.retrouver_g(data_loaded["tab_phi"],rtg.proba_exp(x3(data_loaded["u_conv_tpce"]),5,1001),5),
               5,x3,"g(x)=x^3")

rtg.plot_g_exp(rtg.retrouver_g(data_loaded["tab_phi"],rtg.proba_exp(x3(data_loaded["u_conv_pce"]),5,1001),5),
               5,x3,"g(x)=x^3")

rtg.plot_g_exp(rtg.retrouver_g(data_loaded["tab_phi"],rtg.proba_exp(x3(data_loaded["u_conv_gce"]),5,1001),5),
               5,x3,"g(x)=x^3")
'''

def seuil(x):
    if x>0:
        return(1/2)
    return(-1/2)
seuil_vect = np.vectorize(seuil)
'''
rtg.plot_g_exp(rtg.retrouver_g(data_loaded["tab_phi"],rtg.proba_exp(seuil_vect(data_loaded["u_conv_tpce"]),1,1001),5),
               1,seuil_vect,"seuil")

rtg.plot_g_exp(rtg.retrouver_g(data_loaded["tab_phi"],rtg.proba_exp(seuil_vect(data_loaded["u_conv_pce"]),1,1001),5),
               1,seuil_vect,"seuil")

rtg.plot_g_exp(rtg.retrouver_g(data_loaded["tab_phi"],rtg.proba_exp(seuil_vect(data_loaded["u_conv_gce"]),1,1001),5),
               1,seuil_vect,"seuil")
'''

#arctan
'''
rtg.plot_g_exp(rtg.retrouver_g(data_loaded["tab_phi"],rtg.proba_exp(np.arctan(data_loaded["u_conv_tpce"]),2,1001),5),
               2,np.arctan,"arctan")

rtg.plot_g_exp(rtg.retrouver_g(data_loaded["tab_phi"],rtg.proba_exp(np.arctan(data_loaded["u_conv_pce"]),2,1001),5),
               2,np.arctan,"arctan")

rtg.plot_g_exp(rtg.retrouver_g(data_loaded["tab_phi"],rtg.proba_exp(np.arctan(data_loaded["u_conv_gce"]),2,1001),5),
               2,np.arctan,"arctan")
'''
#U_exp = rtg.inverser_g(rtg.retrouver_g(data_loaded["tab_phi"],rtg.proba_exp(np.arctan(data_loaded["u_conv_pce"]),5,10001),5),
#                       np.arctan(data_loaded["u_conv_pce"]),5)

#k_exp = mp.retrouver(U_exp)[0]

#mp.printimage([U_exp,data_loaded["u_conv_pce"],fftshift(k_exp),fftshift(data_loaded["k_cer_petit_conv_norme"])],
#              ["exp","vrai","k_exp","k0"])

rtg.retrouve_tot(np.arctan(data_loaded["u_conv_pce"]))

'''
data = {
    "wh": ex.wh,
    "k_carre_petit_conv_norme": ex.k_carre_petit_conv_norme,
    "u_conv_pca": ex.u_conv_pca,
    "k_carre_grand_conv_norme": ex.k_carre_grand_conv_norme,
    "u_conv_gca": ex.u_conv_gca,
    "k_cer_tres_petit_conv_norme":ex.k_cer_tres_petit_conv_norme,
    "u_conv_tpce":ex.u_conv_tpce,
    "k_cer_petit_conv_norme":ex.k_cer_petit_conv_norme,
    "u_conv_pce":ex.u_conv_pce,
    "k_cer_grand_conv_norme":ex.k_cer_grand_conv_norme,
    "u_conv_gce":ex.u_conv_gce,
    "k_dirac_conv_norme":ex.k_dirac_conv_norme,
    "u_conv_dirac":ex.u_conv_dirac,
    "tab_phi":rtg.tab_phi(5,1000000)
}

# Écriture (mode binaire)
#with open("data.pkl", "wb") as f:
#    pickle.dump(data, f)
'''

