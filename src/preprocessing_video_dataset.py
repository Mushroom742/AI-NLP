import cv2
import pandas as pd
from os import listdir
import os

''''
Récupère les données du dataset ava et enregistre les images des personnes
présentes dans les vidéos
'''
def save_frames(dataset):
    #défintion des chemins d'accès et de sauvegarde en séparant les dataset
    if dataset == 'train':
        folder_path = '../data/' + 'ava_activespeaker_train_v1.0'
        save_path = '../data/train_frames'
    elif dataset == 'test':
        folder_path = '../data/' + 'ava_activespeaker_test_v1.0'
        save_path = '../data/test_frames'
    
    #lecture des csv de chaque vidéo
    for file in listdir(folder_path):
        data = pd.read_csv(folder_path + '/' + file)
        
        #récupération de l'identifiant de la vidéo
        video_name = data.iat[0,0]
        print(video_name)
        
        #création du dossier
        dir_path = save_path + '/' + video_name
        os.mkdir(dir_path)
        
        #récupération de l'extension
        if os.path.exists('../data/ava_dataset/' + video_name + '.mp4'):
            video_path = '../data/ava_dataset/' + video_name + '.mp4'
        elif os.path.exists('../data/ava_dataset/' + video_name + '.mkv'):
            video_path = '../data/ava_dataset/' + video_name + '.mkv'
        elif os.path.exists('../data/ava_dataset/' + video_name + '.webm'):
            video_path = '../data/ava_dataset/' + video_name + '.webm'
            
        #ouverture de la vidéo
        cap = cv2.VideoCapture(video_path)
        
        #index de la ligne lue
        lign = 0
        
        #pour chaque ligne du fichier csv, récupération du frame_timestamp
        for frame_timestamp in data.iloc[:,1]:
            
            #positionnement de la vidéo sur le frame_timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC,frame_timestamp * 1000)
            
            #récupération de l'image
            ret, frame = cap.read()
            
            #conversion en gris
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            #récupération de la fenêtre représentant le visage
            x1 = int(data.iloc[lign,2] * cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            y1 = int(data.iloc[lign,3] * cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            x2 = int(data.iloc[lign,4] * cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            y2 = int(data.iloc[lign,5] * cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            person_frame = gray_frame[y1:y2, x1:x2]
            
            #sauvegarde de l'image
            cv2.imwrite(dir_path + '/' + str(lign) + '.jpg',\
                        person_frame)
                
            #incrémentation de la ligne lue
            lign += 1
            
        #fermeture de la vidéo
        cap.release()    
                    
        
        
if __name__ == "__main__":
    print("train set")
    save_frames("train")
    print("test set")
    save_frames("test")