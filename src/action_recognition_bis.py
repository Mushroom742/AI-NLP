import pandas as pd
import numpy as np
from tensorflow.keras.layers import TimeDistributed, Conv2D, Dense, MaxPooling2D, Flatten, LSTM, Dropout, BatchNormalization
from tensorflow.keras import models
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from PIL import Image
import os.path
import cv2


'''
Associe un label à chaque image
0 --> la personne ne parle pas
1 --> la personne parle
'''
def create_labels():
    #création des dictionnaires
    partition = {'train': [], 'validation': []}
    labels = {}

    #pour chaque type de dataset
    for dataset in ['train', 'validation']:

        #récupération du chemin du dossier des fichiers csv en fonction du dataset
        if dataset ==  'train':
            folder_path = '../data/' + 'ava_activespeaker_train_v1.0'
            files = ['_a9SWtcaNj8-activespeaker.csv', '_mAfwH6i90E-activespeaker.csv']
        elif dataset == 'validation':
            folder_path = '../data/' + 'ava_activespeaker_test_v1.0'
            files = ['_7oWZq_s_Sk-activespeaker.csv']

        #pour chaque fichier csv
        for file in files:
            data = pd.read_csv(folder_path + '/' + file)

            #récupération de l'identifiant de la vidéo
            video_name = data.iat[0,0]

            #index de la ligne lue
            lign = 0

            #récupération de l'action de chaque image
            for action in data.iloc[:,6]:

                #définition du label en fonction de l'action
                if action == 'NOT_SPEAKING':
                    label = 0
                elif action == 'SPEAKING_AUDIBLE' or action == 'SPEAKING_NOT_AUDIBLE':
                    label = 1

                #association du label à l'image correspondante
                labels[video_name + str(lign)] = label

                #ajout du nom de l'image dans la bonne partition
                partition[dataset].append(video_name + str(lign))

                #incrémentation de la ligne lue
                lign += 1

    return partition, labels

'''
Génération des données des images groupées
'''
def preprocessing_images(list_IDs, labels, images_path, nb_frames=10,\
                     dim=(256,256), n_channels=1):

    #création de X et y
    X = []
    y = []

    #index de la liste
    i = 0
    stop = int(len(list_IDs) / nb_frames) * nb_frames

    #tant que la liste n'est pas parcourue
    while i < stop:

        #initialisation
        frames_path = []
        action = 0

        #récupération des chemins et du label de chaque groupement d'image
        for j in range(nb_frames):
            if i + j < len(list_IDs):

                #récupération des infos sur l'ID (nom de vidéo et numéro de frame)
                video_name = list_IDs[i + j][:11]
                num_frame = list_IDs[i + j][11:]

                frames_path.append(video_name + '/' + num_frame + '.jpg')
                action += labels[list_IDs[i + j]]

        #initialisation
        frames = []

        #récupration des datas associées à chaque image du groupement
        for frame in frames_path:
            with Image.open(images_path + '/' + frame) as image:
                image = image.resize(dim, Image.ANTIALIAS)
                image_np = np.asarray(image)
                image_np_norm = image_np / 255
                frames.append(image_np_norm)

        #transformation en array + ajout à X
        frames = np.array(frames)
        X.append(frames)

        #si au moins une personne parle dans le groupement, label SPEAKING
        if action == 0:
            y.append(0)
        else:
            y.append(1)

        #incrémentation du compteur
        i += nb_frames

    #transformation en array
    X = np.array(X)
    X = X.reshape((int(len(list_IDs) / nb_frames), nb_frames, *dim, n_channels))
    y = np.array(y)

    return X, y

'''
Retourne une liste de prédictions pour chaque visage d'une liste d'images
1 si la personne parle
0 sinon
'''
def predict(frames, faces, nb_frames=10, dim=(256,256)):
    predictions = []

    #chargement du modèle
    if os.path.exists("../results/model_action_recognition"):
        model = models.load_model("../results/model_action_recognition")
    else:
        model = main()

    i = 0
    frames_tmp = []

    #pour chaque image et visage
    for frame, face in zip(frames, faces):
        if face != []:

            #groupe formé
            if i >= nb_frames:

                #prediction
                frames_tmp = np.array(frames_tmp)
                prediction = int(model.predict(frames_tmp).round().item())
                for range(nb_frames):
                    predictions.append(prediction)

                #reset
                i = 0
                frames_tmp = []

            #préparation des données
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            person_frame = gray_frame[face[0][1]:face[0][3], face[0][0]:face[0][2]]
            resize_frame = cv2.resize(person_frame, dim)
            frame_np = np.asarray(resize_frame)
            frame_np_norm = frame_np / 255
            frames_tmp.append(frame_np_norm)

    return predictions


'''
Graphique de la précison du modèle
'''
def plot_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Précision du modèle')
    plt.ylabel('Précision')
    plt.xlabel('Répétitions')
    plt.legend(['entraînement', 'validation'], loc='upper left')
    #plt.show()
    plt.savefig("../results/action_recognition_accuracy.png")
    plt.close()

'''
Graphique de la perte du modèle
'''
def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Erreur du modèle')
    plt.ylabel('Erreur')
    plt.xlabel('Répétitions')
    plt.legend(['entraînement', 'validation'], loc='upper left')
    #plt.show()
    plt.savefig("../results/action_recognition_loss.png")
    plt.close()

'''
Création du modèle + entrainement
'''
def main():
    #création des labels et partition
    partition, labels = create_labels()

    #paramètres
    batch_size = 10
    dim = (256,256)
    nb_frames = 10
    n_channels = 1

    input_shape = (nb_frames, *dim, n_channels)

    #génération des données
    train_data, train_labels = preprocessing_images(partition['train'], labels, \
                    images_path='../data/train_frames', dim=dim, \
                    nb_frames=nb_frames, n_channels=n_channels)
    val_data, val_labels = preprocessing_images(partition['validation'], labels, \
                    images_path='../data/test_frames', dim=dim, \
                    nb_frames=nb_frames, n_channels=n_channels)

    #création du modèle
    model = models.Sequential()
    model.add(TimeDistributed(Conv2D(128, (3, 3), strides=(1, 1), \
                                activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(2, 2)))
    model.add(TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu')))
    model.add(TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(2, 2)))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Flatten()))
    model.add(Dropout(0.2))

    model.add(LSTM(32, return_sequences=False, dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    #compilation du modèle
    optimizer = optimizers.RMSprop(lr=0.01)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    #entrainement du modèle
    history = model.fit(train_data, train_labels, batch_size=batch_size, epochs=5, \
                        validation_data=(val_data, val_labels))

    #sauvegarde du modèle
    model.save("../results/model_action_recognition")

    #graphiques
    plot_accuracy(history)
    plot_loss(history)

    return model

if __name__ == '__main__':
    main()

