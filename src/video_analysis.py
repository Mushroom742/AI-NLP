import speech_recognition
import sys
import os.path
import sentiment_analysis
import cvlib as cv
import cv2
from ffpyplayer.player import MediaPlayer
import textwrap
from unidecode import unidecode

'''
Formate la transcription du texte pour en faire des sous titres
Découpage en morceaux de 50 caractères + conversion de l'unicode en ASCII (sans accents)
'''
def format_subtitles(text):
    new_text = []
    for sent in text:
        new_text.append(textwrap.wrap(unidecode(sent), width=50))
    return new_text

'''
Prend en paramètre un chemin vers une vidéo, analyse les sentiments de l'audio,
et détermine si la personne sur l'image est en train de parler ou non
'''
if __name__ == '__main__':
    #vérification des arguments
    if len(sys.argv) < 2 or not os.path.isfile(sys.argv[1]):
        raise Exception('usage: python video_analysis.py chemin_fichier_video')

    #conversion du fichier audio
    audio_file = speech_recognition.convert_to_wav(sys.argv[1])

    #transcription de l'audio
    timestamp, text = speech_recognition.speech_to_text(audio_file)

    #analyse des sentiments
    sentiments = sentiment_analysis.main(text)

    #formatage des sous titres
    sub_text = format_subtitles(text)

    #ouverture de la vidéo
    cap = cv2.VideoCapture(sys.argv[1])

    #récupération du nombre d'images par seconde
    fps = cap.get(cv2.CAP_PROP_FPS)

    #récupération de la hauteur de la vidéo
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #listes pour récupérer les images et la reconnaissance des visages
    frames = []
    faces = []

    #lecture de la vidéo
    while(cap.isOpened()):

        #récupération de l'image
        ret, frame = cap.read()
        frames.append(frame)
        if not ret:
            break

        #reconnaissance des visages sur l'image
        face, confidences = cv.detect_face(frame)
        faces.append(face)

    #fermeture de la vidéo
    cap.release()

        #analyse mouvement

    #intervalle entre chaque image pour la lecture
    interval = int(1000/fps)

    #lecteur audio
    player = MediaPlayer(sys.argv[1])

    #compteur pour les sous titres
    i = 0
    len_timestamp = len(timestamp)

    #police des sous titres
    font = cv2.FONT_HERSHEY_PLAIN

    #lecture de chaque image
    for num_frame, (frame, face) in enumerate(zip(frames, faces)):

        #rectangle bleu autour de chaque visage
        cv2.rectangle(frame, (face[0][0], face[0][1]), (face[0][2], face[0][3]), (255,0,0), thickness=3)

        #temps actuel en secondes
        time = int(num_frame/fps)

        #comparaison du temps actuel et du temps de début des sous titres
        if(i < len_timestamp and time >= timestamp[i]):

            #passage au sous titre suivant
            if((i + 1) < len_timestamp and time >= timestamp[i + 1]):
                i = i + 1

            #couleur des sous titres en fonction du sentiment
            if (sentiments[i] == 1):

                #positif -> vert
                color = (0,255,0)
            else:

                #négatif -> rouge
                color = (0,0,255)

            #hauteur des sous titres
            y = height - 50

            #affichage des sous titres sur plusieurs lignes avec 15 pixels d'écart
            for sent in sub_text[i]:
                cv2.putText(frame, sent, (50, y), font, 1, color, 1, cv2.LINE_AA)
                y += 15

        #lecture de l'audio
        audio_frame, val = player.get_frame()

        #lecture de l'image
        cv2.imshow('Video analysis', frame)

        #attente entre chaque image ou 'q' pour arrêter
        if cv2.waitKey(interval) == ord('q'):
            break

        #vérification que l'audio n'est pas fini
        if val != 'eof' and audio_frame is not None:
            img, t = audio_frame

    #destruction des fenêtres
    cv2.destroyAllWindows()


