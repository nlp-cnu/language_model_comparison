#import glob
#import os
#import pickle
#import torch

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['TFHUB_CACHE_DIR'] = 'C:\debug'
#from sklearn.model_selection import train_test_split
#import tensorflow as tf
#import tensorflow_hub as hub
#import tensorflow_addons as tfa
#import tensorflow_text as text
#from official.nlp import optimization
#from tensorflow.keras.models import load_model
#from tensorflow.keras.callbacks import *
#from transformers import AutoTokenizer, AutoModel, TFAutoModel
#import numpy as np
#from tensorflow.keras import Model, Sequential
#from sklearn.metrics import classification_report, f1_score
#from tensorflow.keras.layers import *
#from sklearn.model_selection import StratifiedKFold

from Classifier import *
from Dataset import *
import sys


#This is the main running method for the script
if __name__ == '__main__':
    cmdargs = sys.argv
    print(cmdargs)
    mn = cmdargs[1]
    print('model chosen:', mn)
    lr = float(cmdargs[2])
    print('learning rate:', lr)
    back_prop = cmdargs[3]
    #hard-coded variables
    language_model_name = ""
    if mn == 'BASEBERT':
        language_model_name = Classifier.BASEBERT
    elif mn == 'DISTILBERT':
        language_model_name = Classifier.DISTILBERT
    elif mn == 'ROBERTA':
        language_model_name = Classifier.ROBERTA
    elif mn == 'GPT2':
        language_model_name = Classifier.GPT2
    elif mn == 'ALBERT':
        language_model_name = Classifier.ALBERT
    data_filepath = '../text_classification_dataset.tsv'
    seed = 2005

    if back_prop:
        metric_file = "metrics/{}_{}_{}_metrics.txt".format(mn, lr, "BP")
    else:
        metric_file = "metrics/{}_{}_{}_metrics.txt".format(mn, lr, "noBP")
    print('writing metrics to:', metric_file)
    with open(metric_file, 'w') as file:
        file.write('epoch,loss,macro_precision,macro_recall,macro_f1,micro_precision,micro_recall,micro_f1\n')

    #create classifier and load data for a binary text classifier
    #classifier = Binary_Text_Classifier(language_model_name)
    #data = Binary_Text_Classification_Dataset(data_filepath)

    #create classifier and load data for a multiclass text classifier
    num_classes = 2
    classifier = MultiLabel_Text_Classifier(language_model_name, num_classes, rate=lr, metric_file=metric_file,
                                            language_model_trainable=back_prop)
    data = MultiClass_Text_Classification_Dataset(data_filepath)
    
    #get the training data
    train_x, train_y = data.get_train_data()

    #get test data
    test_x, test_y = data.get_test_data()

    ###### BONUS STUFF ########
    #summarize the model in text
    classifier.model.summary()
    #plot the model (an image)
    tf.keras.utils.plot_model(
        classifier.model,
        to_file="model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )
    
    #train the model
    classifier.train(train_x,train_y)

    #predict with the model
    predictions = classifier.test(test_x)

    #TODO - compute test statistics ---- or output the predictions to file or something


    
    

   
