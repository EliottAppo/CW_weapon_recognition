

from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, Toplevel
from tkinter.filedialog import*
import os
import cv2
from Fonctions.prediction import plot_preds, prediction
from Fonctions.fonctions_intermediaires_interface.arme_detectee import arme_est_detectee
from Fonctions.fonctions_intermediaires_interface.envoi_mail import envoie
from Fonctions.fonctions_intermediaires_interface.arme_absente import arme_est_absente
from Fonctions.lecture_video import capture_video
from Fonctions.redimensionnement import redimensionnement

window = Tk()
window.geometry("643x356")
window.configure(bg="#1E116F")



def recherche():  # permet d'acceder au chemin de la photo/video à analyser
    
    filepath = askopenfilename(title="Ouvrir une image", filetypes=[
                               ('all files', '.*')])  # chemin du fichier à analyser
    window.destroy()
    seuil = 0.2  # probabilité de présence à dépasser

    if filepath[-4:] == '.mp4':
        capture_video(filepath,seuil)

    else:

        preds = prediction(filepath)
        # stocke le nom de l'arme avec la plus grande probabilité de présence
        detected_gun = list(preds.keys())[0]

        # si la probabilité de présence d'une arme dépasse un certain seuil
        if preds[detected_gun] > seuil:
            # envoie()                       #pour potentiellement envoyer un mail
            cv2.imshow("image importée", cv2.imread(filepath))
            arme_est_detectee(detected_gun)
            plot_preds(filepath, preds)  # affiche un diagramme de probabilité

        else:
            arme_est_absente()
            plot_preds(filepath, preds)


def webcam(): #Fonction pour la webcam
    seuil = 0.2
    window.destroy()
    capture_video(0, seuil)



# création de l'interface


def creation_interface():

    # interface vierge
    canvas = Canvas(
        window,
        bg="#1E116F",
        height=356,
        width=643,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )
    # rectangle noir en bas de l'initerface
    canvas.place(x=0, y=0)
    canvas.create_rectangle(
        0.0,
        322.97265625,
        643.0,
        356.0,
        fill="#180001",
        outline="")

    # image en haut à gauche
    image_image_1 = PhotoImage(
        file="Fonctions/fonctions_intermediaires_interface/image_interface/image_1.png")
    image_1 = canvas.create_image(
        126.0,
        89.0,
        image=image_image_1
    )

    # bouton "déposer un fichier"
    button_image_1 = PhotoImage(
        file="Fonctions/fonctions_intermediaires_interface/image_interface/button_1.png")
    button_1 = Button(
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=recherche
    )
    button_1.place(
        x=299.0,
        y=215.0,
        width=314.0,
        height=58.0
    )
    button_image_3 = PhotoImage(
        file="Fonctions/fonctions_intermediaires_interface/image_interface/button_3.png")
    button_3 = Button(
        image=button_image_3,
        borderwidth=0,
        highlightthickness=0,
        command=webcam
    )
    button_3.place(
        x=600.0,
        y=325.0,
        width=30.0,
        height=30.0
    )
    canvas.create_text(
    500.0,
    325.0,
    anchor="nw",
    text="Webcam",
    fill="#FFFFFF",
    font=("RobotoSlab Regular", 23 * -1)
    )

    # rectangle derrière le texte
    canvas.create_rectangle(
        321.0,
        65.0,
        585.0,
        182.0,
        fill="#181D23",
        outline="")

    # image en bas à gauche
    image_image_2 = PhotoImage(
        file="Fonctions/fonctions_intermediaires_interface/image_interface/image_2.png")
    image_2 = canvas.create_image(
        126.0,
        244.0,
        image=image_image_2
    )

    # texte "Reconnaissance d'armes à feu"
    canvas.create_text(
        370.0,
        100.0,
        anchor="nw",
        text="Reconnaissance \n   d'armes à feu",
        fill="#FFFFFF",
        font=("RobotoSlab Regular", 23 * -1)
    )


    window.resizable(False, False)
    window.mainloop()
