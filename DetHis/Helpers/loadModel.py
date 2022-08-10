import tensorflow as tf
from tensorflow import keras
import pickle


def getPerceptron():
    perceptron = tf.keras.models.load_model('.\DetHis\ModelosEntrenados\perceptron')
    #perceptron.summary()
    return perceptron

def getConvolutional():
    convolutional = tf.keras.models.load_model('.\DetHis\ModelosEntrenados\convolutional')
    #convolutional.summary()
    return convolutional

def getKNN():
    with open('.\DetHis\ModelosEntrenados\knn.pkl', 'rb') as f:
        knn = pickle.load(f)
    return knn