import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
import sklearn.model_selection
from abc import ABC, abstractmethod

# import preprocessor as p
# from sklearn.utils import class_weight

SEED = 3


# Abstract dataset class
class Dataset(ABC):

    @abstractmethod
    def __init__(self, seed=SEED, validation_set_size=0):  # use_all_data=False,
        # validation_set_size is the percentage to use for validation set (e.g. 0.2 = 20%
        self.seed = seed
        self._val_set_size = validation_set_size
        self._train_X = None
        self._train_Y = None
        self._val_X = None
        self._val_Y = None

    # I should also maintain the class ratio during the val/train split...is that happening here?
    # .... I should check the sklearn implementation
    def _training_validation_split(self, data, labels):
        # Split data
        if (self._val_set_size >= 1):
            raise Exception("Error: test set size must be greater than 0 and less than 1")
        if (self._val_set_size > 0):
            self._train_X, self._val_X, self._train_Y, self._val_Y = sklearn.model_selection.train_test_split(data,
                                                                                                              labels,
                                                                                                              test_size=self._val_set_size,
                                                                                                              random_state=self.seed)
        else:
            self._train_X = data
            self._train_Y = labels
            self._val_X = None
            self._val_Y = None

    def get_train_data(self):
        if self._train_X is None or self._train_Y is None:
            raise Exception(
                "Error: train data does not exist, you must call _training_validation_split after loading data")
        return self._train_X, self._train_Y

    def get_validation_data(self):
        if self._val_X is None or self._val_Y is None:
            raise Exception("Error: val data does not exist, you must specify a validation split percent")
        return self._val_X, self._val_Y

    # You can tweek this however you want
    def preprocess_data(self, data):
        # preprocess tweets to remove mentions, URL's
        p.set_options(p.OPT.MENTION, p.OPT.URL)  # P.OPT.HASHTAG, p.OPT.EMOJI
        data = data.apply(p.clean)

        # Tokenize special Tweet characters
        # p.set_options(p.OPT.NUMBER)  # p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.RESERVED,
        # data = data.apply(p.tokenize)

        return data.tolist()

    def get_train_class_weights(self):
        return self.class_weights

    def _determine_class_weights(self):
        # determine class weights
        self.class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self._train_Y),
            y=self._train_Y
            # y=self._train_Y.argmax(axis=1) #TODO -- use this (or something like it) for multiclass problems
        )
        self.class_weights = dict(enumerate(self.class_weights))


#TODO - implement multi-label text classification dataset loader

# Loads data in which there is a single (categorical) label column (e.g. class 0 = 0, class 2 = 2)
class MultiClass_Text_Classification_Dataset(Dataset):
    def __init__(self, data_file_path, text_column_name=None, label_column_name=None, seed=SEED, validation_set_size=0):
        Dataset.__init__(self, seed=seed, validation_set_size=validation_set_size)
        # load the labels

        if (text_column_name is None or label_column_name is None):
            text_column_name = 'text'
            label_column_name = 'label'
            df = pd.read_csv(data_file_path, header=None, names=[text_column_name, label_column_name],
                             delimiter='\t').dropna()
        else:
            df = pd.read_csv(data_file_path, delimiter='\t').dropna()

        label_encoder = OneHotEncoder(sparse=False)
        labels = label_encoder.fit_transform(df[label_column_name].values.reshape(-1, 1))

        # load the data
        raw_data = df[text_column_name]
        data = df[text_column_name].values.tolist()
        # data = self.preprocess_data(raw_data)

        self._training_validation_split(data, labels)
        # self._determine_class_weights() #TODO - re-implement this
    
#Load a data and labels for a text classification dataset
class Binary_Text_Classification_Dataset(Dataset):
    '''
    Class to load and store a text classification dataset. Text classification datasets
    contain text and a label for the text, and possibly other information. Columns
    are assumed to be tab seperated and each row corresponds to a different sample

    Inherits from the Dataset class, only difference is in how the data is loaded
    upon initialization
    '''
    def __init__(self, data_file_path, text_column_name=None, label_column_name=None, label_column_names=None, seed=SEED, test_set_size=0):
        '''
        Method to instantiate a text classification dataset
        :param data_file_path: the path to the file containing the data
        :param text_column_name: the column name containing the text
        :param label_column_name: the column name containing the label (class) of the text (for binary labeled data)
        :param label_column_names: a list of column names containing the label (class) of the text data (for multi-label data)
           text_column_name must be specified for multi-label data
        :param seed: the seed for random split between test and training sets
        :param make_test_train_split: a number between 0 and 1 determining the percentage size of the test set
           if no number is passed in (or if a 0 is passed in), then no test train split is made
        '''
        Dataset.__init__(self, seed=seed, test_set_size=test_set_size)

        #load the labels
        if(label_column_names is None): #check if binary or multi-label #TODO --- this isn't necessarily true. I think I should just load the data differently for multilabel or multi-class problems (create a different method)
            #no label column names passed in, so it must be binary
            #load the labels with or without column header info
            if (text_column_name is None or label_column_name is None):
                text_column_name = 'text'
                label_column_name = 'label'
                df = pd.read_csv(data_file_path, header=None, names=[text_column_name, label_column_name], delimiter='\t').dropna()
            else:
                df = pd.read_csv(data_file_path, delimiter='\t').dropna()

            labels = df[label_column_name].values.reshape(-1, 1)
            print ("labels = ", labels)
        
        #load multilabel classification data. Column names are required
        else:
            #TODO - implement this if/when it is needed. Actual implementation depends on the data format
            if (text_column_name is None):
                raise Exception("Error: text_column_name must be specified to load multilabel data")

            #NOTE: in the case of multiclass data, where class is an integer, it is easy to encode
            # as one-hot data with the following command:
            #label_encoder = OneHotEncoder(sparse=False)
            #labels = label_encoder.fit_transform(df[label_column_name].values.reshape(-1, 1))
            #----however, with a single column this else statement won't get called since its just a single column
            #    ...so, would need to modify this method
            
            #sklearn.multilabel binarizer is another option.
            # in the end though, what you should get is a list of lists e.g. [0,1,1],[1,0,0],[0,1,0] where each triplet
            # are the labels for a single sample. If confised, check the label encoder to check
            raise Exception("Error: not yet implemented, depends on dataset")

            
        #load the data
        raw_data = df[text_column_name]
        data = df[text_column_name].values.tolist()
        #data = self.preprocess_data(raw_data)
        
        self._test_train_split(data, labels)
        #self._determine_class_weights() #TODO - re-implement this
        

#TODO -- may need to expand for multi-class problems
#TODO -- may need to expand for different format types. Right now it is for span start and span end format types
class Token_Classification_Dataset(Dataset):
    def __init__(self, data_file_path, text_column_name, span_start_column_name, span_end_column_name, tokenizer, seed=SEED, test_set_size=0):
        Dataset.__init__(self, seed=seed, test_set_size=test_set_size)
        self.tokenizer = tokenizer

        # read in data
        df = pd.read_csv(data_file_path, delimiter='\t').dropna()
        data = self.preprocess_data(df['text'])
        span_start = df[start]
        span_end = df[end]

        #tokenize the data to generate y-values for each token
        model_name = TODO_PASS_IN
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenized = tokenizer(batch_x, padding=True, truncation=True, max_length=512, return_tensors='tf')


        for i,t in enumerate(tokenized['input_ids']):
            print("string = ", batch_x[i])
            print("tokens = ", self.tokenizer.convert_ids_to_tokens(t))
            print("attention_mask = ", tokenized['attention_mask'][i])
        
        labels = TODO ####NEED TO TOKENIZE THE DATA and ADD LABELS ACCORDINGLY

        #TODO - need a label mapper or something. This is the tricky part, but once I get labels passed into the model I think
        # I can go ahead and run the model I created for it
        
        self._test_train_split(data, labels)
        self._determine_class_weights()
