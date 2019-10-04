from keras.applications.inception_v3 import preprocess_input
from keras.applications import imagenet_utils



def getPreprocessForModel(modelName):
    if modelName == 'inception' or modelName == 'xception':
        return preprocess_input
    return imagenet_utils.preprocess_input
