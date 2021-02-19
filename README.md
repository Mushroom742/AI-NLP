# AI-NLP
Analyse de scènes audio/vidéo à travers le traitement du langage naturel et le traitement d'images

## Librairies à installer
`
pip install vosk
pip install pandas
pip install numpy
pip install tensorflow
pip install matplotlib
pip install opencv-pyhton
pip install cvlib
pip install ffpyplayer
pip install unidecode
pip install pillow
`

## Démonstration
Pour lancer la démonstration, se placer dans le dossier `src` et lancer la commande `python video_analysis.py` **`chemin_de_la_vidéo`**
Pour pouvoir lancer la démonstration, il faut que les 2 modèles soient entrainés et sauvegardés.

## Entrainement du modèle de classification des émotions
Télécharger https://www.kaggle.com/hbaflast/french-twitter-sentiment-analysis et le placer dans le dossier `data`
Télécharger un des modèles sur https://alphacephei.com/vosk/models pour la transcription et l'extraire dans `src/model`
Dans le dossier `src`, lancer la commande `python sentiment_analysis.py`, le modèle sera sauvegardé sous `results/model_sentiment_analysis`

## Entrainement du modèle de classification des actions
Télécharger https://research.google.com/ava/download/ava_activespeaker_train_v1.0.tar.bz2 et https://research.google.com/ava/download/ava_activespeaker_val_v1.0.tar.bz2 et les extraire dans le dossier `data`
Télécharger https://s3.amazonaws.com/ava-dataset/annotations/ava_speech_file_names_v1.txt et le placer dans `data`
Dans le dossier `src`, lancer la commande `python dl_ava_dataset.py` et placer les vidéos téléchargées dans `data/ava_dataset`
Dans le dossier `src`, lancer la commande `python preprocessing_video_dataset.py`, pour récupérer les images
Dans le dossier `src`, lancer la commande `python action_recognition.py`, le modèle sera sauvegardé sous `results/model_action_recognition`
