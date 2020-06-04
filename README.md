# THreat dEtection by moving bOunDaries around nORmal sAmple (THEODORA)

The repository contains code refered to the work:

_Giuseppina Andresini, Annalisa Appice, Francesco Paolo Caforio,  Donato Malerba_

[Improving Cyber-Threat Detection by Movingthe Boundary around the Normal Samples]() 

Please cite our work if you find it useful for your research and work.
```
  @ARTICLE{}
```




## Code requirements

The code relies on the following **python3.6+** libs.

Packages need are:
* [Tensorflow 1.13](https://www.tensorflow.org/) 
* [Keras 2.3](https://github.com/keras-team/keras) 
* [Matplotlib 2.2](https://matplotlib.org/)
* [Pandas 0.23.4](https://pandas.pydata.org/)
* [Numpy 1.15.4](https://www.numpy.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)

## Data
The datasets used for experiments are accessible from [__DATASETS__](https://drive.google.com/open?id=1OIfsMv2PJljkc0aco00WB4_t8gEnXMiE). Original dataset is transformed in a binary classification: "_threat_, _normal_" (_oneCls files).
The repository contains the orginal dataset (folder: "original") and  the dataset after the preprocessing phase (folder: "numeric") 

Preprocessing phase is done mapping categorical feature and performing the Min Max scaler.

## How to use
Repository contains scripts of all experiments included in the paper:
* __main.py__ : script to run THEODORA
* __mindful.py__ : script to run MINDFUL model 

  
 Code contains models and datasets used for experiments in the work.
 
  

## Replicate the experiments

To replicate experiments reported in the work, you can use models and datasets stored in homonym folders.
Global variables are stored in __THEODORA.conf__  file 


```python
    N_CLASSES = 2
    PREPROCESSING1 = 0  #if set to 1 code execute preprocessing phase on original date
    LOAD_AUTOENCODER_ADV = 1 #if 1 the autoencoder for attacks items  is loaded from models folder
    LOAD_AUTOENCODER_NORMAL = 1 #if 1 the autoencoder for normal items  is loaded from models folder
    LOAD_CNN = 1  #if 1 the classifier is loaded from models folder
    VALIDATION_SPLIT #the percentage of validation set used to train models
    CHANGE_CLASS_SVC = 1 #if set to 1 the boundary re-positiong is performed
    LOAD_SVC = 1 #if 1 the SVM model for decision boundary is load
    THRESHOLD = 0.70 #threshold for change of normal class
```

## Download datasets

[All datasets](https://drive.google.com/drive/folders/1OIfsMv2PJljkc0aco00WB4_t8gEnXMiE?usp=sharing)

All models and plots performed for the experiment about **Decision boundary re-positioning** can be dowloaded [here](https://drive.google.com/drive/folders/1ap0p4uYqljU5BvWQZAQqryfQvzwFN6i7)
