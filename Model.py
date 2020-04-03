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


from sklearn.metrics import confusion_matrix


np.set_printoptions(suppress=True)
from sklearn.svm import SVC
import pickle
import matplotlib.pyplot as plt
import collections, numpy

from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
import Utils

from Mindful import Mindful


class Run():
    def __init__(self, dsConfig, config, dataset):
        self.config = config
        self.ds = dsConfig
        self.dataset= dataset

    def run(self):
        print('THEODORA EXECUTION')

        dsConf = self.ds
        pathModels = dsConf.get('pathModels')
        pathPlot = dsConf.get('pathPlot')
        configuration = self.config


        N_CLASSES = int(configuration.get('N_CLASSES'))
        pd.set_option('display.expand_frame_repr', False)

        # contains path of dataset and model and preprocessing phases
        ds = Datasets(dsConf)
        ds.preprocessing1()
        if(self.dataset=='CICIDS2017'):
            train, test = ds.getTrain_TestCIDIS()
        else:
            train, test = ds.getTrain_Test()
        prp = prep(train, test)

        # Preprocessing phase from original to numerical dataset
        PREPROCESSING1 = int(configuration.get('PREPROCESSING1'))
        if (PREPROCESSING1 == 1):

            train, test = ds.preprocessing2(prp)
        else:
            train, test = ds.getNumericDatasets()

        clsT, clsTest = prp.getCls()

        if (self.dataset == 'CICIDS2017'):
            train_X, train_Y, test_X, test_Y = prp.getXYCICIDS(train, test)
        else:
            train_X, train_Y, test_X, test_Y = prp.getXY(train, test)



        change_class_svc = int(configuration.get('CHANGE_CLASS_SVC'))
        load_model_svc = int(configuration.get('LOAD_SVC'))
        plot_cluster = int(configuration.get('PLOT_CLUSTER'))

        '''Boundary detection phase'''
        if (change_class_svc == 1):

            if (load_model_svc == 0):

                print('Training')
                kernel=configuration.get('kernel')
                gamma=configuration.get('gamma')
                c = configuration.get('C')
                decision = configuration.get('decision_function_shape')
                if kernel is None:
                    classifier_conf = SVC(probability=True)
                else:
                    classifier_conf = SVC(probability=True, kernel=kernel, gamma=gamma, C=c,
                                          decision_function_shape=decision)
                classifier_conf.fit(train_X, train_Y)
                # save the model to disk
                pickle.dump(classifier_conf, open(pathModels + 'svc-model.sav', 'wb'))

            else:

                print('Loading')
                # load the model from disk
                classifier_conf = pickle.load(open(pathModels + 'svc-model.sav', 'rb'))

            print('Prediction SVC')
            prob = classifier_conf.predict_proba(train_X)
            #Probabilty to prection as attacks
            prob0 = prob[:,0]
            prob1 = prob[:, 1]

            # SVC prediction
            pred=[0 if p >0.5 else 1 for p in prob0]

            cluster0 = pred.count(0)
            cluster1=pred.count(1)

            print('Number of elements in cluster0 (attack): ', cluster0)
            print('Number of elements in cluster1 (normal): ', cluster1)


            true_Y = train_Y[clsT].tolist()
            cf=confusion_matrix(true_Y, pred)


            print("Number of elements of class 0 clustered as 0: ", cf[0][0])
            print("Number of elements of class 1 clustered as 1: ", cf[1][1])
            print("Number of elements of class 1 clustered as 0: ", cf[1][0])
            print("Number of elements of class 0 clustered as 1: ", cf[0][1])

            purity = (1 / (len(true_Y)) * (cf[0][0] + cf[1][1]))
            print('Purity Index: ', purity)

            df_svc = train_Y.copy()
            df_svc['Prob1'] = prob1




            print(collections.Counter(train_Y[clsT]))


            threshold = float(configuration.get('THRESHOLD'))

            df_selected= df_svc.loc[(df_svc[clsT]  == 1) & (df_svc["Prob1"] >= 0.50) & (df_svc["Prob1"] <= threshold)]
            df_svc.loc[df_selected.index, clsT] = 0

            print('#LabelChanged: ', df_selected.shape[0])
            print('Threshold - Changing classes: ', threshold)

            train[clsT] = df_svc[clsT]

            print(collections.Counter(train[clsT]))

            # Plot
            if (plot_cluster == 1):
                print('Plot - Cluster')
                Plot.plotCluster(df_svc, pathPlot, clsT)

            plot_cluster_change = int(configuration.get('PLOT_CLUSTER1_CHANGING'))
            if (plot_cluster_change == 1):
                print('Plot - Cluster')
                Plot.plotClusterChange(df_selected, pathPlot, threshold)



        # create pandas for results
        columns = ['TP', 'FN', 'FP', 'TN', 'OA', 'AA', 'P', 'R', 'F1', 'FAR(FPR)', 'TPR']
        results = pd.DataFrame(columns=columns)


        ''' Classification phase'''
        print('Start classification phase')
        mdf = Mindful(self.ds, self.config, clsT, train)
        mdf.run()



        '''Predition'''
        if (self.dataset == 'CICIDS2017'):
            r_list = []
            i = 0
            test_X_image = list()
            for t in test:
                t_i = mdf.createDS(t)
                test_X_image.append(t_i)

            for t, Y in zip(test_X_image, test_Y):
                i += 1
                predictionsC = mdf.prediction(t)
                print('Softmax on test set')
                y_pred = np.argmax(predictionsC, axis=1)
                cm = confusion_matrix(Y, y_pred)
                print(cm)
                r = Utils.getResult(cm, N_CLASSES)
                r_list.append(tuple(r))

            dfResults_temp = pd.DataFrame(r_list, columns=columns)
            drMean = dfResults_temp.mean(axis=0)
            drmeanList = pd.Series(drMean).values
            r_mean = []
            for i in drmeanList:
                r_mean.append(i)

            dfResults = pd.DataFrame([r], columns=columns)
            print(dfResults)


        else:
            test_X_image = mdf.createDS(test)
            prediction = mdf.prediction(test_X_image)
            y_pred = np.argmax(prediction, axis=1)
            cm = confusion_matrix(test_Y, y_pred)
            print('Prediction Test')
            print(cm)

            r = Utils.getResult(cm, N_CLASSES)

            dfResults = pd.DataFrame([r], columns=columns)
            print(dfResults)

        results = results.append(dfResults, ignore_index=True)

        results.to_csv(ds._testpath + '_results.csv', index=False)
