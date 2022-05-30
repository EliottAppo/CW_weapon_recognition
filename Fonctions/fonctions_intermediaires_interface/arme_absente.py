import tkinter 
from tkinter import *

def arme_est_absente():  
    print("Rien à signaler")
    fen = Tk()                            #création d'une fenêtre pour informer l'utilisateur 
    can = Canvas(fen, width=430, height=300, bg='white')
    can.pack(side='top', fill='both', expand='yes')
    photo = PhotoImage(file="Fonctions/fonctions_intermediaires_interface/image_interface/RAS.png") #importation d'une image à placer dans la fenêtre
    can.create_image(0,0,anchor='nw', image=photo, tag='photo')
    fen.mainloop()