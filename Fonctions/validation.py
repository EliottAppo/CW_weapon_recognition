from prediction import prediction
import os 
from tensorflow.keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt

def validation(pathbdd,seuil):
    '''Cette fonction permet de calculer le taux de bonne prévision sur une base de données'''
    taux = 0
    nb = 0
    for dossier in os.listdir(pathbdd):
        for filename in os.listdir(os.path.join(pathbdd,dossier)):# on parcourt tous les fichiers

            dict = prediction(os.path.join(pathbdd,os.path.join(dossier,filename))) # probabilité d'apparition des 3 types d'armes (classé dans l'ordre croissant de proba)
            nb+=1
            detected_gun = list(dict.keys())[0]
            if dict[detected_gun]>seuil:# si la probabilité de présence est supérieure au seuil
                taux += (dossier==detected_gun or (detected_gun=="assault_rifle" and dossier=="rifle") ) # si l'arme détectée est la bonne #on n'a que 2 catégories dans la base de données (rifle et revolver), on confondra donc assault_rifle et rifle
            else:
                taux+=(dossier=="autre") # si le modèle n'a rien détecté et qu'il n'y avait effectivement pas d'armes
    taux = taux/nb
    return taux

def seuil_variable(pathbdd,Liste_seuil):
    '''Cette fonction permet de générer la liste des taux de bonne prévision obtenus avec chaque seuil de Liste_indice
    dans l'objectif de représenter le taux de bonne prévision en fonction du seuil'''
    Liste_resultats = []
    for x in Liste_seuil:
        y = validation(pathbdd,x)
        Liste_resultats.append(y)
    return Liste_resultats


Liste_seuil = [0.1 +k*0.1 for k in range(9)]
Liste_taux = [0.8224181360201511,0.792191435768262, 0.7682619647355163, 0.7531486146095718,0.707808564231738,0.6335012594458438, 0.5516372795969773, 0.4760705289672544, 0.37531486146095716] # résultat de seuil_variable("BDD",Liste_seuil), déjà écrite car long à exécuter

def plot_seuil(Liste_seuil,Liste_taux):
    '''Affiche le graphe Liste_y = f(Liste_seuil)'''
    plt.plot(Liste_seuil,Liste_taux)
    plt.ylabel("Taux de bonne prévision")
    plt.xlabel("Seuil de probabilité choisi")
    plt.title("Taux de bonne prévision en fonction du seuil")
    plt.legend()
    plt.show()

#plot_seuil(Liste_seuil,Liste_taux)
