from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
import cv2
###Utilisation du modèle ResNet50
modèle1 = ResNet50(weights='imagenet')


def traitement_preds(preds_model,ensemble):
    '''Cette fonction prend en entrée la prédiction du modèle ResNet50 contenant les probabilités de présence de 1000 objets
    pour la traiter et ne conserver que celles qui nous intéressent (la présence des objets de ensemble)'''
    L = decode_predictions(preds_model, top=1000)[0]
    preds ={}
    for (x,y,z) in L:
        if y in ensemble:
            preds[y]=z
    return preds




def prediction(img_path,modèle=modèle1,ensemble=set(["revolver","rifle","assault_rifle"])):
    ''' Cette fonction permet de charger une image et de renvoyer les probabilités d'apparition des différents objets de ensemble 
    Sortie : dictionnaire de la forme {revolver:0.8 , rifle : 0.55 , assault_rifle : 0.2}'''
    
    img = image.load_img(img_path, target_size=(224, 224))
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds_model = modèle.predict(x)
    preds = traitement_preds(preds_model,ensemble)
    return preds

def prediction_image(img_path,modèle=modèle1,ensemble=set(["revolver","rifle","assault_rifle"])):
    ''' Cette fonction permet de charger une image et de renvoyer les probabilités d'apparition des différents objets de ensemble 
    Sortie : dictionnaire de la forme {revolver:0.8 , rifle : 0.55 , assault_rifle : 0.2}'''
    
    img = cv2.resize(img_path, (224, 224), interpolation=cv2.INTER_CUBIC)
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds_model = modèle.predict(x)
    preds = traitement_preds(preds_model,ensemble)
    return preds




def plot_preds(img_path, preds_model):  
    """Permet d'afficher un histogramme reprenant les probabilités d'apparition de chaque arme à feu sur la photo"""
    
    #img = image.load_img(img_path, target_size=(224, 224))
    #plt.imshow(img)                                           #affiche l'image
    order = list(reversed(range(len(preds_model.keys()))))
    plt.figure()  
    bar_preds = list(preds_model.values())
    labels = list(preds_model.keys())
    plt.title("Probabilité de présence de chaque arme")
    plt.barh(order,bar_preds, alpha=0.5)
    plt.yticks(order,labels = labels)
    plt.xlabel('Probabilité')
    plt.xlim(0, 1.01)
    plt.tight_layout()
    plt.show()


#preds = prediction('BDD/revolver/1.jpg')
#detected_gun = list(preds.keys())[0]
#print(detected_gun)