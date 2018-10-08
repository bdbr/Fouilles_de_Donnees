import numpy as np

def parseBD_music(ref,film):
    """Parse le fichier csv ref.
    film est le numero de colonne du film
    dont on veut prédie les notes. 
    """
    pairs = []
    bd = open(ref,"r").readlines()
    name_film =(bd[0].split(",")[film+4])
    print(name_film)
    for i in range(1,len(bd)):
        pairs.append((bd[i].split(",")[film],bd[i].split(",")[1:19]))
    return pairs
        
        
def transform_data(pairs):
    """Transforme les donnee de pairs en
    np.array
    Si une note est manquante on la remplace par 3 
    And so on... 
    """
    y = np.array([int(f) if f !='' else 3 for (f,_) in pairs])
    x = np.array([m for (_,m) in pairs])
    for t in range(len(x)):
        x[t] = [int(e) if e != '' else 3 for e in x[t]]
    return x.astype(int),y.astype(int)
