from django.core.files.storage import FileSystemStorage
import numpy as np
import librosa


def generateMFCCArray(path, max_len=1000):
    array = []
    signal, sr = librosa.load(path, mono=True, sr=None)
    mfcc = librosa.feature.mfcc(y=signal, n_mfcc=20, sr=sr)
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=(
            (0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    array.append(mfcc)
    array = np.array(array)
    array.reshape(array.shape[0], 20, 1000, 1)
    return array


def saveFile(fileName, file):
    storage = FileSystemStorage(location="DetHis/apiFiles/")
    file_name = storage.save(fileName, file)
    return storage.path(file_name)


def deleteFile(fileName):
    storage = FileSystemStorage(location="DetHis/apiFiles/")
    storage.delete(fileName)


def getAcentos():
    return np.load("DetHis/Helpers/acentos.npy")
