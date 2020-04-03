import configparser
import sys
import numpy as np

np.random.seed(12)
import tensorflow
tensorflow.set_random_seed(12)
import Model





def datasetException():
    try:
        dataset=sys.argv[1]

        if (dataset is None) :
            raise Exception()
        if not ((dataset == 'KDDCUP99') or (dataset == 'CICIDS2017') or (dataset=='AAGM')):
            raise ValueError()
    except Exception:
        print("The name of dataset is null: use KDDCUP99 or CICIDS2017 or AAGM")
    except ValueError:
        print ("Dataset not exist: must be KDDCUP99 or CICIDS2017 or AAGM")
    return dataset





def main():

    dataset=datasetException()

    config = configparser.ConfigParser()
    config.read('THEODORA.conf')

    dsConf = config[dataset]
    configuration = config['setting']

    execution=Model.Run(dsConf,configuration, dataset)
    execution.run()



if __name__ == "__main__":
    main()