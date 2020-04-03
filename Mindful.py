import numpy as np

my_seed = 12
np.random.seed(my_seed)
import random

random.seed(my_seed)


import tensorflow

tensorflow.set_random_seed(my_seed)

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from Preprocessing import Preprocessing as prep
from DatasetsConfig import Datasets
import Plot
from keras import callbacks
from keras.utils import np_utils

from sklearn.metrics import confusion_matrix
from keras.models import Model

from keras.models import load_model
from keras import backend as K
from keras.utils import plot_model
np.set_printoptions(suppress=True)
from sklearn.svm import SVC
import pickle
import matplotlib.pyplot as plt
import collections, numpy
import Utils as ut


from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


class Mindful():
    def __init__(self, dsConfig, config,cls, train):
        self.configuration = config
        self.dsConfig = dsConfig
        self.cls=cls
        # contains path of dataset and model and preprocessing phases
        self.ds = Datasets(dsConfig)
        self.pathModels = self.dsConfig.get('pathModels')
        self.pathPlot = self.dsConfig.get('pathPlot')
        self.TrainImage =0
        self.TestImage=0
        self.model=0
        self.autoencoderN=0
        self.autoencoderA=0
        self.train=train





    def createImage(self, train_X, trainA, trainN):
        '''Create MINDFUL image based on the recosntruction error
        of Autoencoder Nomrmal and Autoencoder Attacks'''
        rows = [train_X, trainA, trainN]
        rows = [list(i) for i in zip(*rows)]

        train_X = np.array(rows)

        if K.image_data_format() == 'channels_first':
            x_train = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2])
            input_shape = (train_X.shape[1], train_X.shape[2])
        else:
            x_train = train_X.reshape(train_X.shape[0], train_X.shape[2], train_X.shape[1])
            input_shape = (train_X.shape[2], train_X.shape[1])
        return x_train, input_shape


    def split_Normal_Attack(self, train):
        '''Split dataset in samples of class normal and attacks'''
        train_normal = train[(train[self.cls] == 1)]
        train_anormal = train[(train[self.cls] == 0)]
        return train_normal, train_anormal


    def learnAutoencoderNormal(self, train_XN, N_CLASSES, VALIDATION_SPLIT):
        ''' Learning of autoencoder trained on normal samples'''
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, restore_best_weights=True),
        ]

        print('LOAD_AUTOENCODER_NORMAL')
        autoencoderN, p = self.ds.getAutoencoder_Normal(train_XN, N_CLASSES)
        autoencoderN.summary()

        history = autoencoderN.fit(train_XN, train_XN,
                                   validation_split=VALIDATION_SPLIT,
                                   batch_size=p['batch_size'],
                                   epochs=p['epochs'], shuffle=True,
                                   callbacks=callbacks_list,
                                   verbose=1)
        autoencoderN.save(self.pathModels + 'autoencoderNormal.h5')
        Plot.printPlotLoss(history, 'autoencoderN', self.pathPlot)
        return autoencoderN

    def learnAutoencoderAttacks(self, train_XA, N_CLASSES, VALIDATION_SPLIT):
        ''' Learning of autoencoder trained on normal samples'''
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, restore_best_weights=True),
        ]

        print('LOAD_AUTOENCODER_NORMAL')
        autoencoderA, p = self.ds.getAutoencoder_Attacks(train_XA, N_CLASSES)
        autoencoderA.summary()

        history = autoencoderA.fit(train_XA, train_XA,
                                   validation_split=VALIDATION_SPLIT,
                                   batch_size=p['batch_size'],
                                   epochs=p['epochs'], shuffle=True,
                                   callbacks=callbacks_list,
                                   verbose=1)
        autoencoderA.save(self.pathModels + 'autoencoderNormal.h5')
        Plot.printPlotLoss(history, 'autoencoderA', self.pathPlot)
        return autoencoderA


    def learnCNN(self,train_X, train_Y, N_CLASSES, VALIDATION_SPLIT, input_shape):
        ''' Learning 1DCNN'''
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5,
                                    restore_best_weights=True),
        ]

        model, p = self.ds.getMINDFUL(input_shape, N_CLASSES)

        history3 = model.fit(train_X, train_Y,
                             # validation_data=(test_X, test_Y2),
                             validation_split=VALIDATION_SPLIT,
                             batch_size=p['batch_size'],
                             epochs=p['epochs'], shuffle=True,  # shuffle=false for NSL-KDD true for UNSW-NB15
                             callbacks=callbacks_list,  # class_weight=class_weight,
                             verbose=1)

        Plot.printPlotAccuracy(history3, 'finalModel1', self.pathPlot)
        Plot.printPlotLoss(history3, 'finalModel1', self.pathPlot)
        model.save(self.pathModels + 'MINDFUL.h5')

        return model


    def getModel(self):
        ''' Return the classifier'''
        return self.model


    def getAutoencoderN(self):
        ''' Return the autoencoder trained on normal samples'''
        return self.autoencoderN

    def getAutoencoderA(self):
        ''' Return the autoencoder trained on attacks samples'''
        return self.autoencoderA


    def run(self):
        ''' Run the learning stage of MINDFUL'''
        train_normal, train_anormal=self.split_Normal_Attack(self.train)


        train_XN, train_YN = ut.getXY(train_normal,  self.cls)
        train_XA, train_YA= ut.getXY(train_anormal, self.cls)

        train_X, train_Y = ut.getXY(self.train, self.cls)

        print('Train data shape normal', train_XN.shape)
        print('Train target shape normal', train_YN.shape)


        print('Train data shape anormal', train_XA.shape)
        print('Train target shape anormal', train_YA.shape)


        # convert class vectors to binary class matrices fo softmax
        train_Y2 = np_utils.to_categorical(train_Y, int(self.configuration.get('N_CLASSES')))
        print("Target train shape after", train_Y2.shape)
        print("Train all", train_X.shape)


        N_CLASSES = int(self.configuration.get('N_CLASSES'))
        VALIDATION_SPLIT = float(self.configuration.get('VALIDATION_SPLIT'))

        if (int(self.configuration.get('LOAD_AUTOENCODER_NORMAL')) == 0):
            autoencoderN=self.learnAutoencoderNormal(train_XN, N_CLASSES, VALIDATION_SPLIT)
        else:

            print("Load autoencoder Normal from disk")
            autoencoderN = load_model(self.pathModels + 'autoencoderNormal.h5')
            autoencoderN.summary()

        self.autoencoderN =autoencoderN

        if (int(self.configuration.get('LOAD_AUTOENCODER_ADV')) == 0):
            autoencoderA = self.learnAutoencoderAttacks(train_XA, N_CLASSES, VALIDATION_SPLIT)

        else:
            print("Load autoencoder Attacks from disk")
            autoencoderA = load_model(self.pathModels + 'autoencoderAttacks.h5')
            autoencoderA.summary()

        self.autoencoderA =autoencoderA

        train_RE = autoencoderN.predict(train_X)


        train_REA = autoencoderA.predict(train_X)


        train_X_image, input_Shape = self.createImage(train_X, train_RE, train_REA)  # XS UNSW


        self.TrainImage= train_X_image


        if (int(self.configuration.get('LOAD_CNN')) == 0):

            model=self.learnCNN(train_X_image, train_Y2, N_CLASSES, VALIDATION_SPLIT, input_Shape)
        else:
            print("Load softmax from disk")
            model = load_model(self.pathModels + 'MINDFUL.h5')
            model.summary()

        self.model=model

    def createDS(self,test):
        ''' Mapp a dataset on image created using autoencoders'''
        test_normal, test_anormal = self.split_Normal_Attack(test)
        test_XN, test_YN = ut.getXY(test_normal,  self.cls)
        test_XA, test_YA = ut.getXY(test_anormal, self.cls)
        test_X, test_Y = ut.getXY(test, self.cls)

        print('Test data shape normal', test_XN.shape)
        print('Test target shape normal', test_YN.shape)
        print('Test data shape anormal', test_XA.shape)
        print('Test target shape anormal', test_YA.shape)
        print("Test all", test_X.shape)
        autoencoderN = self.getAutoencoderN()
        test_RE = autoencoderN.predict(test_X)

        autoencoderA = self.getAutoencoderA()
        test_REA = autoencoderA.predict(test_X)

        test_X_image, input_shape = self.createImage(test_X, test_RE, test_REA)
        return test_X_image

    def prediction(self, test_X_image):
        '''Use of MINDFUL for prediction on image'''
        model = self.getModel()

        predictions = model.predict(test_X_image)

        return predictions






