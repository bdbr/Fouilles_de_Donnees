from traitementDonnee import *
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import ttest_ind
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def regression_lineaire(x,y):
    """
    Divise x et y en un ensemble de test et d'entrainement
    utilise l'ensemble d'entrainement pour trouver 
    un model par regression lineaire et utilise l'ensemble de test pour
    vérifier l'efficacité du model
    return le mae du model
    """
    reg = linear_model.LinearRegression()
    mus_train, mus_test, flm_train, flm_test = train_test_split(x, y)
    reg.fit(mus_train, flm_train)
    Xr = reg.predict(mus_test)
    plt.scatter(Xr, flm_test, c='r',marker='x')
    plt.savefig("../doc/sca_regress.png")
    return mean_absolute_error(Xr,flm_test)

def resultat_regres(ref):
    """
    Fait appel à regression_lineaire pour predire 
    la note de chaque les film 
    """
    mae =[]
    for n in range(20,31):
        film = n
        pairs = parseBD_music(ref,film)
        (x,y)=transform_data(pairs)
        mae.append(regression_lineaire(x,y))
        print(mae[n-20])
    print("Moyenne: ",sum(mae)/len(mae))
    plt.figure()
    plt.bar(range(len(mae)), mae)
    plt.savefig("../doc/bar_regress.png")
    plt.show()

print("le resultat de regression lineaire:")
print("=============================================================")
resultat_regres("../donnee/responses.csv")
