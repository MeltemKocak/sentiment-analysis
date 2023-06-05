import numpy as np

import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

#  degiskenler private  olsun
__model = None
__tokenizer = None


def making_prediction(sentence):
    # degiskenler global yapama
    global __model
    global __tokenizer
    # tokenizer yukleme
    with open("tweeter_tokenizer.pickle", "rb") as f:
        __tokenizer = pickle.load(f)
        print(__model)

    # model yukleme
    __model2 = tf.keras.models.load_model("tweetter.h5")
    # gelen cumle list donusturme
    sentence = [sentence]
    # gelen cumle sequence alma
    sequences = __tokenizer.texts_to_sequences(sentence)
    # gelen cumle post padding yapma
    padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
    # tahmin etme
    prediction = __model2.predict(padded)
    return prediction


if __name__ == '__main__':
    prd = making_prediction("didn’ write chowkidar does mean ’ anti modi try visit the plz not all who haven’ used are anti ")
    enyuksek = np.argmax(prd)
    print(enyuksek)
