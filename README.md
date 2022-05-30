# Projet

![Screenshot](/Fonctions/fonctions_intermediaires_interface/image_interface/Interface_image.jpeg)

Notre projet consiste à détecter les armes à feu sur des vidéos pour créer des alertes de sécurité. 

## Initialisation

Voici les recommandations pour installer et lancer notre programme sur votre machine dans des conditions optimales.

### Prérequis

Ce que vous avez à installer et comment le faire 

Python

Modules nécessaires :
* tensorflow
* numpy
* opencv
* tkinter
* smtplib
* fiftyone
* pytest

### Installation

Utiliser pip pour installer les modules requis

```
pip install -r requirements.txt
```


## Comment l'utiliser 

Lancer la fonction main et donner le chemin de votre fichier
* Si une arme est detectée, une alerte sera envoyée par mail (à modifier par l'utilsateur) et un pop-up apparaitra
* Sinon une alerte "pas d'arme detectée" est affichée

## Authors
* Eliott Dumont
* Mathis Phan
* Dac-An Vo
* Elian Mangin
* Augustin Pagniez
* Luc Bernard

## Remarque 
La base de données utilisé pour l'entrainement est disponible en envoyant un mail à eliott.dumont@student-cs.fr
