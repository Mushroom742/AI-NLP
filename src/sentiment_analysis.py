import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D, Embedding
import tensorflow.keras.models
import matplotlib.pyplot as plt
import os.path

'''
Transforme les données textes en associant un numéro à chaque mot (dans l'ordre 
des mots les plus utilisés du dataset).

Renvoie les textes encodés dans des vecteurs de tailles égales, le nombre de 
mots du dataset et la taille des vecteurs.
'''
def preprocessing_data(texts, num_words=5000):

    #séparation et normalisation des mots (pas de ponctuation ou de majuscules)
    #+ association d'un nombre à chaque mot
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)

    #nombre de mots différents
    vocab_size = len(tokenizer.word_index) + 1

    #remplacement de chaque mot par son nombre
    sequences = tokenizer.texts_to_sequences(texts)

    #ajout de 0 en début de chaque vecteur pour avoir la même taille que le 
    #plus grand
    padded_sequences = pad_sequences(sequences)

    return (tokenizer,padded_sequences, vocab_size, len(padded_sequences[0]))

def plot_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Précision du modèle')
    plt.ylabel('Précision')
    plt.xlabel('Répétitions')
    plt.legend(['entraînement', 'validation'], loc='upper left')
    #plt.show()
    plt.savefig("../results/sentiment_analysis_accuracy.png")
    plt.close()

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Erreur du modèle')
    plt.ylabel('Erreur')
    plt.xlabel('Répétitions')
    plt.legend(['entraînement', 'validation'], loc='upper left')
    #plt.show()
    plt.savefig("../results/sentiment_analysis_loss.png")
    plt.close()
    
def main(sentences):
    #ouverture du fichier et mélange des données
    data = pd.read_csv("../data/french_tweets.csv")
    data = data.sample(frac=1).reset_index(drop=True)

    #récupération des données textes
    texts = data.text.values

    #mise en forme des données
    tokenizer, padded_sequences, vocab_size,\
        input_length = preprocessing_data(texts)
        
    if not os.path.exists("../results/model_sentiment_analysis"):
        #construction du modèle
        embedding_vector_length = 32
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_vector_length,\
                            input_length = input_length))
        model.add(SpatialDropout1D(0.25))
        model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
    
        model.compile(loss='binary_crossentropy', optimizer='adam',\
                      metrics=['accuracy'])
        print(model.summary())
    
        #entrainement du modèle
        history = model.fit(padded_sequences, data['label'], validation_split=0.2,\
                            epochs=5, batch_size=32)
    
        print(model.summary())
    
        #sauvegarde du modele
        model.save("../results/model_sentiment_analysis")
    
        #graphique
        plot_accuracy(history)
        plot_loss(history)
        
    else:
        model = tensorflow.keras.models.load_model('../results/model_sentiment_analysis')

    
    #prédictions
    predictions = []
    
    # test_pos = "Aujourd'hui il fait beau, cela me donne envie de sortir et de faire la fête"
    # test_neg = "Aujourd'hui il fait tout gris et moche, cela me donne envie de rester sous la couette et de ne rien faire"
    
    for sent in sentences:
        seq = pad_sequences(tokenizer.texts_to_sequences([sent]),\
                            maxlen=input_length)
        predictions.append(int(model.predict(seq).round().item()))
        #print(sent)
        # if prediction == 1:
        #     print('positif')
        # else:
        #     print('négatif')
    return predictions

if __name__ == "__main__":
    main([])