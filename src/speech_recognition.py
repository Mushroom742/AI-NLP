from vosk import Model, KaldiRecognizer, SetLogLevel
import wave
import json
import os
import subprocess

'''
Convertis un fichier audio en format WAV PCM 16khz 16bit mono

Nécessite l'installation de ffmpeg
'''
def convert_to_wav(in_file):
    file_name, file_ext = os.path.splitext(in_file)
    out_file = file_name + "1.wav"

    process = subprocess.check_call(['ffmpeg', '-i', in_file, '-bitexact', '-y','-ar', '16000', '-ac', '1', out_file])


'''
Retranscris un fichier audio au format WAV PCM 16khz 16bit mono en texte de
la forme suivante : [ horodatage ] retranscription

Nécessite d'avoir les données pour l'entrainement du modèle dans le dossier
courant (https://alphacephei.com/vosk/models à dézipper dans le dossier model)
'''
def speech_to_text(file):
    SetLogLevel(0)

    #ouverture fichier audio
    with wave.open(file, "rb") as wav_file:

        #entrainement modèle avec les données
        model = Model("model")
        rec = KaldiRecognizer(model, wav_file.getframerate())

        #lecture du fichier audio par bloc de frames
        data = wav_file.readframes(4000)
        while len(data) != 0:
            if rec.AcceptWaveform(data):
                #récupération de la transcription par json
                res = json.loads(rec.Result())
                #affichage de la transcription avec l'horodatage
                print('[ ',res['result'][0]['start'],' ] ',res['text'])

            #lecture du bloc de frames suivant
            data = wav_file.readframes(4000)

if __name__ == "__main__":
    file = '../data/test/Macron.wav'
    convert_to_wav(file)
    speech_to_text('../data/test/Macron1.wav')