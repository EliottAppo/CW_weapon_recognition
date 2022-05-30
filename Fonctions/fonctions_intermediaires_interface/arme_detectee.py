import tkinter 
from tkinter import *


def arme_est_detectee(type_weapon): ##fait apparaître une alerte si une arme est détectée
    print("Be careful! A {} has been detected!".format(type_weapon))
    fen = Tk()
    can = Canvas(fen, width=490, height=300, bg='white')
    can.pack(side='top', fill='both', expand='yes')
    photo = PhotoImage(file="Fonctions/fonctions_intermediaires_interface/image_interface/hautlesmains.png") #importation d'une image à placer dans la fenêtre
    can.create_image(0,0,anchor='nw', image=photo, tag='photo')
    fen.mainloop()


