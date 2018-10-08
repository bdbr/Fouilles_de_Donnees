from traitementDonnee import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


def random_forest(x,y):
    """
    Divise x et y en un ensemble de test et d'entrainement
    utilise l'ensemble d'entrainement pour trouver 
    un model a l'aide de l'algorithm des random forest et utilise l'ensemble de test pour
    vérifier l'efficacité du model
    return le mae du model
    """
    model = RandomForestRegressor(n_estimators = 100,
     max_features = "sqrt")
    mus_train, mus_test, flm_train, flm_test = train_test_split(x, y)
    model.fit(mus_train, flm_train)
    predictions = model.predict(mus_test)
    plt.scatter(predictions, flm_test, c='r',marker='x')
    plt.savefig("../doc/sca_forest.png")
    return mean_absolute_error(predictions,flm_test)

def resultat_forest(ref):
    """
    Fait appel à regression_lineaire pour predire 
    la note de chaque les film 
    retourne la moyenne des mae.
    """
    mae =[]
    for n in range(20,31):
            film = n
            pairs = parseBD_music(ref,film)
            (x,y)=transform_data(pairs)
            mae.append(random_forest(x,y))
            print(mae[n-20])
    print("Moyenne: ",sum(mae)/len(mae))
    plt.figure()
    plt.bar(range(len(mae)), mae)
    plt.savefig("../doc/bar_forest.png")
    plt.show()
    return sum(mae)/len(mae)

print("resultat de random forest:")
print("=============================================================")
resultat_forest("../donnee/responses.csv")
