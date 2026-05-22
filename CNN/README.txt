Pour que le code fonctionne, vous devez installer les librairies suivantes.
Il est recommandé d'utiliser un environnement virtuel.

Créez un fichier "requirements.txt" et mettez-y les lignes suivantes :
---------------------------
ultralytics
opencv-python
numpy
customtkinter
Pillow
PyYAML
---------------------------
Téléchargez les bibliothèques à l'aide de ce code dans le terminal. : py -m pip install numpy customtkinter ultralytics opencv-python pillow 

---------------------------------------------------------------------------------------
!!!!! Pour utiliser le modèle Yolo entraîné, rendez-vous dans le dossier final. !!!!!!!
---------------------------------------------------------------------------------------

Assurez-vous que votre modèle entraîné se trouve bien dans le dossier :
`train/weights/best.pt`


1. Lancez le script `main_app.py` :
   python main_app.py

---------------------------------------------------------------------------------------
Si vous souhaitez tester chaque script, rendez-vous dans le dossier : « test ». 
---------------------------------------------------------------------------------------



---------------------------------
  3. ENTRAINEMENT (Optionnel)
---------------------------------

Si vous voulez entraîner votre propre modèle :

Organisez votre dataset comme il faut (dossiers `images` et `labels`).

!!! Dans le dossier dataset, vous trouverez un fichier d'annotation pour étiqueter les images.



Lancez le script `train.py` :
   python train.py


