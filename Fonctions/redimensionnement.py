import numpy as np
import cv2
def redimensionnement(img, h): #réduit/agrandit l'image en gardant 
                               #le meme rapport longueur/hauteur (nouvelle hauteur =h)
    
    hauteur, longueur, m = np.shape(img)
    rapport = longueur/hauteur
    l=h * rapport
    return cv2.resize(img, (h, l), interpolation=cv2.INTER_CUBIC)


#print(redimensionnement(cv2.imread('C:/Users/mathi/Documents/reco/project_semaine2_gp6/images_tests/beau gun.jpg'), 400))

