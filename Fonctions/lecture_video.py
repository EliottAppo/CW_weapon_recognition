import numpy
import cv2
from Fonctions.prediction import prediction_image
from Fonctions.fonctions_intermediaires_interface.arme_detectee import arme_est_detectee


def capture_video(file_path, seuil):
    #	0 = webcam , 'path/nom_video.mp4' = video enregistrée
    cap = cv2.VideoCapture(file_path)
    arret = False
    # boucle infinie (peut être mettre la fonction de détection dans cette boucle, attention à ne pas prendre toutes les frames!)
    while arret ==False:
        n = 0
        while n < 1:
            ret, frame = cap.read()
            n += 1
        # stocker l'image issue de la vidéo à l'instant t dans la variable "frame"
        ret, frame = cap.read()
        if ret == True:
            
            preds = prediction_image(frame)
            # stocke le nom de l'arme avec la plus grande probabilité de présence
            detected_gun = list(preds.keys())[0]

            # si la probabilité de présence d'une arme dépasse un certain seuil
            if preds[detected_gun] > seuil:
                # envoie()                       #pour potentiellement envoyer un mail

                
                arme_est_detectee(detected_gun)
                arret = True
            
            # afficher l'image contenue dans "frame"
            cv2.imshow('output', frame)

        # quitter la boucle infinie lorqu'on appuie sur la touche 'q', waitKey(N), fait attendre N ms.
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    # quiter le programme et fermer toutes les fenêtres ouvertes
    cap.release()
    cv2.destroyAllWindows()


#capture_video('C:/Users/mathi/OneDrive/Images/Pellicule/WIN_20211118_13_30_20_Pro.mp4',0.2)
