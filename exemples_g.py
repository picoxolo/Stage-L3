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
rtg.plot_g_exp(rtg.proba_exp(data_loaded["u_conv_tpce"],10000),rtg.tab_phi(10000),id,"g(x)=x")

rtg.plot_g_exp(rtg.proba_exp(data_loaded["u_conv_pce"],10000),rtg.tab_phi(10000),id,"g(x)=x")

rtg.plot_g_exp(rtg.proba_exp(data_loaded["u_conv_gce"],10000),rtg.tab_phi(10000),id,"g(x)=x")

rtg.retrouve_tot(data_loaded["u_conv_tpce"],5000,data_loaded["k_cer_tres_petit_conv_norme"],
                 id,"g(x)=x",data_loaded["u_conv_tpce"],1)
            
rtg.retrouve_tot(data_loaded["u_conv_pce"],5000,data_loaded["k_cer_petit_conv_norme"],
                 id,"g(x)=x",data_loaded["u_conv_pce"],1)

rtg.retrouve_tot(data_loaded["u_conv_gce"],5000,data_loaded["k_cer_grand_conv_norme"],
                 id,"g(x)=x",data_loaded["u_conv_gce"],1)
"""

def id2(x):
    return(2*x)
'''
rtg.plot_g_exp(rtg.proba_exp(id2(data_loaded["u_conv_tpce"]),5000),rtg.tab_phi(5000),id2,"g(x)=2x")

rtg.plot_g_exp(rtg.proba_exp(id2(data_loaded["u_conv_pce"]),5000),rtg.tab_phi(5000),id2,"g(x)=2x")

rtg.plot_g_exp(rtg.proba_exp(id2(data_loaded["u_conv_gce"]),5000),rtg.tab_phi(5000),id2,"g(x)=2x")

rtg.retrouve_tot(id2(data_loaded["u_conv_tpce"]),5000,data_loaded["k_cer_tres_petit_conv_norme"],
                 id2,"g(x)=2x",data_loaded["u_conv_tpce"],1)
            
rtg.retrouve_tot(id2(data_loaded["u_conv_pce"]),5000,data_loaded["k_cer_petit_conv_norme"],
                 id2,"g(x)=2x",data_loaded["u_conv_pce"],1)

rtg.retrouve_tot(id2(data_loaded["u_conv_gce"]),5000,data_loaded["k_cer_grand_conv_norme"],
                 id2,"g(x)=2x",data_loaded["u_conv_gce"],1)
'''

def x3(x):
    return(x**3)
'''
rtg.plot_g_exp(rtg.proba_exp(x3(data_loaded["u_conv_tpce"]),5000),rtg.tab_phi(5000),x3,"g(x)=x^3")

rtg.plot_g_exp(rtg.proba_exp(x3(data_loaded["u_conv_pce"]),5000),rtg.tab_phi(5000),x3,"g(x)=x^3")

rtg.plot_g_exp(rtg.proba_exp(x3(data_loaded["u_conv_gce"]),5000),rtg.tab_phi(5000),x3,"g(x)=x^3")

rtg.retrouve_tot(x3(data_loaded["u_conv_tpce"]),5000,data_loaded["k_cer_tres_petit_conv_norme"],
                 x3,"g(x)=x^3",data_loaded["u_conv_tpce"],1)

rtg.retrouve_tot(x3(data_loaded["u_conv_pce"]),5000,data_loaded["k_cer_petit_conv_norme"],
                 x3,"g(x)=x^3",data_loaded["u_conv_pce"],1)

rtg.retrouve_tot(x3(data_loaded["u_conv_gce"]),5000,data_loaded["k_cer_grand_conv_norme"],
                 x3,"g(x)=x^3",data_loaded["u_conv_gce"],1)
'''

def seuil(x):
    if x>2.5:
        return(1/2)
    return(-1/2)
seuil_vect = np.vectorize(seuil)
'''
rtg.plot_g_exp(rtg.proba_exp(seuil_vect(data_loaded["u_conv_tpce"]),1000),rtg.tab_phi(1000),seuil_vect,"seuil")

rtg.plot_g_exp(rtg.proba_exp(seuil_vect(data_loaded["u_conv_pce"]),5000),rtg.tab_phi(5000),seuil_vect,"seuil")

rtg.plot_g_exp(rtg.proba_exp(seuil_vect(data_loaded["u_conv_gce"]),5000),rtg.tab_phi(5000),seuil_vect,"seuil")

rtg.retrouve_tot(seuil_vect(data_loaded["u_conv_tpce"]),5000,data_loaded["k_cer_tres_petit_conv_norme"],
                 seuil_vect,"seuil",data_loaded["u_conv_tpce"],1)

rtg.retrouve_tot(seuil_vect(data_loaded["u_conv_pce"]),5000,data_loaded["k_cer_petit_conv_norme"],
                 seuil_vect,"seuil",data_loaded["u_conv_pce"],1)

rtg.retrouve_tot(seuil_vect(data_loaded["u_conv_gce"]),5000,data_loaded["k_cer_grand_conv_norme"],
                 seuil_vect,"seuil",data_loaded["u_conv_gce"],1)
'''

#arctan
'''
rtg.plot_g_exp(rtg.proba_exp(np.arctan(data_loaded["u_conv_tpce"]),5000),rtg.tab_phi(5000),np.arctan,"g(x)=arctan(x)")

rtg.plot_g_exp(rtg.proba_exp(np.arctan(data_loaded["u_conv_pce"]),5000),rtg.tab_phi(5000),np.arctan,"g(x)=arctan(x)")

rtg.plot_g_exp(rtg.proba_exp(np.arctan(data_loaded["u_conv_gce"]),5000),rtg.tab_phi(5000),np.arctan,"g(x)=arctan(x)")

rtg.retrouve_tot(np.arctan(data_loaded["u_conv_tpce"]),5000,data_loaded["k_cer_tres_petit_conv_norme"],
                 np.arctan,"g(x)=actan(x)",data_loaded["u_conv_tpce"],1)

rtg.retrouve_tot(np.arctan(data_loaded["u_conv_pce"]),5000,data_loaded["k_cer_petit_conv_norme"],
                 np.arctan,"g(x)=actan(x)",data_loaded["u_conv_pce"],1)

rtg.retrouve_tot(np.arctan(data_loaded["u_conv_gce"]),5000,data_loaded["k_cer_grand_conv_norme"],
                 np.arctan,"g(x)=actan(x)",data_loaded["u_conv_gce"],1)
'''

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

