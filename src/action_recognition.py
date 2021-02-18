from os import listdir
import pandas as pd
import numpy as np
import tensorflow.keras
from tensorflow.keras.layers import TimeDistributed, Conv2D, Dense, MaxPooling2D, Flatten, LSTM, Dropout, BatchNormalization
from tensorflow.keras import models
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from PIL import Image

'''
Pour lire les données à la volée en faisant des groupes de 10 images
'''
class DataGenerator(tensorflow.keras.utils.Sequence):
    '''
    initialisation des variables

    dim = dimensions de l'image
    batch_size = nombre de données par lot
    nb_frames = nombre d'images par input
    labels = les labels associés aux images
    list_IDs = liste des noms d'image
    n_channels = nombre de canaux
    n_classes = nombre de classes en sortie
    shuffle = mélanger les données après chaque epoch
    images_path = le chemin pour accéder aux images
    '''
    def __init__(self, list_IDs, labels, images_path, batch_size=32, nb_frames=10,\
                    dim=(256,256), n_channels=1, n_classes=2, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size * nb_frames
        self.nb_frames = nb_frames
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.images_path = images_path
        self.on_epoch_end()

    '''
    Ce qu'il faut faire à la fin de chaque epoch
    '''
    def on_epoch_end(self):
        #réinitialisation des indexs
        self.indexes = np.arange(len(self.list_IDs))

        #mélanger les données
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    '''
    Génération des données à la volée
    '''
    def __data_generation(self, list_IDs_temp):

        #création de X et y
        X = []
        y = []

        #index de la liste
        i = 0

        #tant que la liste n'est pas parcourue
        while i < len(list_IDs_temp):

            #récupération des infos sur l'ID (nom de vidéo et numéro de frame)
            video_name = list_IDs_temp[i][:11]
            num_frame = list_IDs_temp[i][11:]

            #initialisation
            frames_path = []
            action = 0

            #récupération des chemins et du label de chaque groupement d'image
            for j in range(self.nb_frames):
                if i + j < len(list_IDs_temp):
                    frames_path.append(video_name + '/' + str(int(num_frame) + j) + '.jpg')
                    action += self.labels[list_IDs_temp[i + j]]

            #initialisation
            frames = []

            #récupration des datas associées à chaque image du groupement
            for frame in frames_path:
                with Image.open(self.images_path + '/' + frame) as image:
                    image = image.resize(self.dim, Image.ANTIALIAS)
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
            i += self.nb_frames

        #transformation en array
        X = np.array(X)
        X = X.reshape((int(self.batch_size / self.nb_frames), self.nb_frames, *self.dim, self.n_channels))
        y = np.array(y)

        #y sous la forme [0 0 0 1 ... 0] en fonction de n_classes
        return X, tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes)

    '''
    Définit le nombre de lot par epoch : nb de données / taille d'un lot
    '''
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    '''
    Génération d'un lot de données
    '''
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y


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
        elif dataset == 'validation':
            folder_path = '../data/' + 'ava_activespeaker_test_v1.0'

        #pour chaque fichier csv
        for file in listdir(folder_path):
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

if __name__ == '__main__':
    #création des labels et partition
    partition, labels = create_labels()

    #paramètres
    batch_size = 3
    dim = (256,256)
    nb_frames = 10
    n_channels = 1
    n_classes = 2

    input_shape = (nb_frames, *dim, n_channels)

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

    #initialisation des générateurs de données
    train_gen = DataGenerator(partition['train'], labels, \
                    images_path='../data/train_frames', batch_size=batch_size,\
                    dim=dim, nb_frames=nb_frames, n_channels=n_channels,\
                    n_classes=n_classes, shuffle=False)

    validation_gen = DataGenerator(partition['validation'], labels, \
                    images_path='../data/test_frames', batch_size=batch_size,\
                    dim=dim, nb_frames=nb_frames, n_channels=n_channels,\
                    n_classes=n_classes, shuffle=False)

    #entrainement du modèle
    history = model.fit_generator(generator=train_gen, \
                                    validation_data=validation_gen, epochs=1)

    #sauvegarde du modèle
    model.save("../results/model_action_recognition")

    #graphiques
    plot_accuracy(history)
    plot_loss(history)

