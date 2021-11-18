from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping
import tensorflow_addons as tfa

from DataGenerator import *
from Metrics import *

mf = ""


class WriteMetrics(Callback):
    global mf

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("At start; log keys: ".format(keys))
        print('GLOBAL FILE TEST:', mf)

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End of epoch {}; log keys;: {}".format(epoch+1, keys))
        print(list(logs.values()))
        vals = list(logs.values())
        print('GLOBAL TEST:', mf)
        with open(mf, 'a') as file:
            file.write("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(epoch+1, vals[0], vals[1], vals[2]
                                                                                      , vals[3], vals[4], vals[5],
                                                                                      vals[6], vals[7]))


class Classifier:
    '''
    Classifier class, which holds a language model and a classifier
    This class can be modified to create whatever architecture you want,
    however it requres the following instance variables:
    self.language_mode_name - this is a passed in variable and it specifies
       which HuggingFace language model to use
    self.tokenizer - this is created, but is just an instance of the tokenizer
       corresponding to the HuggingFace language model you use
    self.model - this is the Keras/Tensor flow classification model. This is
       what you can modify the architecture of, but it must be set

    Upon instantiation, the model is constructed. It should then be trained, and
    the model will then be saved and can be used for prediction.

    Training uses a DataGenerator object, which inherits from a sequence object
    The DataGenerator ensures that data is correctly divided into batches. 
    This could be done manually, but the DataGenerator ensures it is done 
    correctly, and also allows for variable length batches, which massively
    increases the speed of training.
    '''
    # These are some of the HuggingFace Models which you can use
    BASEBERT = 'bert-base-uncased'
    DISTILBERT = 'distilbert-base-uncased'
    ROBERTA = 'roberta-base'
    GPT2 = 'gpt2'
    ALBERT = 'albert-base-v2'

    # some default parameter values
    EPOCHS = 50
    BATCH_SIZE = 20

    def __init__(self):
        '''
        Initializer for a language model. This class should be extended, and
        the model should be built in the constructor. This constructor does
        nothing, since it is an abstract class. In the constructor however
        you must define:
        self.tokenizer 
        self.model
        '''
        self.tokenizer = None
        self.model = None

    def train(self, x, y, batch_size=BATCH_SIZE, validation_data=None, epochs=EPOCHS):
        '''
        Trains the classifier
        :param x: the training data
        :param y: the training labels

        :param batch_size: the batch size
        :param: validation_data: a tuple containing x and y for a validation dataset
                so, validation_data[0] = val_x and validation_data[1] = val_y
        :param: epochs: the number of epochs to train for
        '''

        # create a DataGenerator from the training data
        training_data = DataGenerator(x, y, batch_size, self.tokenizer)

        # generate the validation data (if it exists)
        if validation_data is not None:
            validation_data = DataGenerator(validation_data[0], validation_data[1], batch_size, self.tokenizer)

        # fit the model to the training data
        self.model.fit(
            training_data,
            epochs=epochs,
            validation_data=validation_data,
            verbose=2,
            callbacks=[WriteMetrics(), EarlyStopping(monitor='loss', patience=5)]
        )

    # function to predict using the NN
    def predict(self, x, batch_size=BATCH_SIZE):
        """
        Predicts labels for data
        :param x: data
        :return: predictions
        """
        if not isinstance(x, tf.keras.utils.Sequence):
            # TODO - max length can be longer depending on the model -e.g. Roberta is 768
            tokenized = self.tokenizer(x, padding=True, truncation=True, max_length=512, return_tensors='tf')
            x = (tokenized['input_ids'], tokenized['attention_mask'])

        return self.model.predict(x, batch_size=batch_size)


# Example of how to write custom metrics. This is precision, recall, and f1 scores
# TODO - got these from online (https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model), do they work for multi-class problems too?
from keras import backend as K


# def recall_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall
#
#
# def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision
#
#
# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class Binary_Text_Classifier(Classifier):
    def __init__(self, language_model_name, language_model_trainable=False):
        Classifier.__init__(self)
        self.language_model_name = language_model_name

        # create the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)

        # create the language model
        model_name = self.language_model_name
        language_model = TFAutoModel.from_pretrained(model_name)
        language_model.trainable = language_model_trainable
        # language_model.output_hidden_states = False

        # print the GPUs that tensorflow can find, and enable memory growth.
        # memory growth is something that CJ had to do, but doesn't work for me
        # set memory growth prevents tensor flow from just grabbing all available VRAM
        # physical_devices = tf.config.list_physical_devices('GPU')
        # print (physical_devices)
        # tf.config.experimental.set_memory_growth(physical_devices[0], True)

        # create the model
        # create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")

        # create the embeddings - the 0th index is the last hidden layer
        embeddings = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0]

        # We can create a sentence embedding using the one directly from BERT, or using a biLSTM
        # OR, we can return the sequence from BERT (just don't slice) or the BiLSTM (use retrun_sequences=True)
        # create the sentence embedding layer - using the BERT sentence representation (cls token)
        # sentence_representation_language_model = embeddings[:,0,:]
        # Note: we are slicing because this is a sentence classification task. We only need the cls predictions
        # not the individual words, so just the 0th index in the 3D tensor. Other indices are embeddings for
        # subsequent words in the sequence (http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)

        # Alternatively, we can use a biLSTM to create a sentence representation -- This seems to generally work better
        # create the sentence embedding layer using a biLSTM and BERT token representations
        lstm_size = 128
        biLSTM_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_size))
        sentence_representation_biLSTM = biLSTM_layer(embeddings)

        # now, create some deep layers
        # deep 1
        dense1 = tf.keras.layers.Dense(256, activation='gelu')
        dropout1 = tf.keras.layers.Dropout(.2)
        output1 = dropout1(dense1(sentence_representation_biLSTM))
        # output1 = dropout1(dense1(sentence_representation_language_model))

        # deep 2
        dense2 = tf.keras.layers.Dense(128, activation='gelu')
        dropout2 = tf.keras.layers.Dropout(.2)
        output2 = dropout2(dense2(output1))

        # deep 3
        dense3 = tf.keras.layers.Dense(64, activation='gelu')
        dropout3 = tf.keras.layers.Dropout(.2)
        output3 = dropout3(dense3(output2))

        # softmax
        softmax_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        final_output = softmax_layer(output3)

        # combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])

        # compile the model
        optimizer = tf.keras.optimizers.Adam(lr=0.01)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), precision_m, tf.keras.metrics.Recall(), recall_m,
                     tfa.metrics.F1Score(1), f1_m]  # TODO - get F1 working
            # metrics=['accuracy', tfa.metrics.F1Score(2)] #TODO -add precision and recall
        )


class MultiLabel_Text_Classifier(Classifier):
    def __init__(self, language_model_name, num_classes, rate, metric_file, language_model_trainable=False):
        '''
        This is identical to the Binary_Text_Classifier, except the last layer uses
        a softmax, loss is Categorical Cross Entropy and its output dimension is num_classes
        Also, different metrics are reported.
        You also need to make sure that the class input is the correct dimensionality by
        using Dataset TODO --- need to write a new class?
        '''
        Classifier.__init__(self)
        self.language_model_name = language_model_name
        self.num_classes = num_classes
        self.rate = rate
        self.metric_file = metric_file

        # set global variables for name and rate for file writing purposes
        global mf
        mf = metric_file

        # create the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)

        # create the language model
        model_name = self.language_model_name
        language_model = TFAutoModel.from_pretrained(model_name)
        language_model.trainable = language_model_trainable
        # language_model.output_hidden_states = False

        # print the GPUs that tensorflow can find, and enable memory growth.
        # memory growth is something that CJ had to do, but doesn't work for me
        # set memory growth prevents tensor flow from just grabbing all available VRAM
        # physical_devices = tf.config.list_physical_devices('GPU')
        # print (physical_devices)
        # tf.config.experimental.set_memory_growth(physical_devices[0], True)

        # create the model
        # create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")

        # create the embeddings - the 0th index is the last hidden layer
        embeddings = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0]

        # We can create a sentence embedding using the one directly from BERT, or using a biLSTM
        # OR, we can return the sequence from BERT (just don't slice) or the BiLSTM (use retrun_sequences=True)
        # create the sentence embedding layer - using the BERT sentence representation (cls token)
        # embeddings = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0][:,0,:]
        sentence_representation_language_model = embeddings[:, 0, :]
        # Note: we are slicing because this is a sentence classification task. We only need the cls predictions
        # not the individual words, so just the 0th index in the 3D tensor. Other indices are embeddings for
        # subsequent words in the sequence (http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)

        # Alternatively, we can use a biLSTM to create a sentence representation -- This seems to generally work better
        # create the sentence embedding layer using a biLSTM and BERT token representations
        # NOTE: for some reason this (a slice to a biLSTM) throws an error with numpy version >= 1.2
        # but, it works with numpy 1.19.5
        lstm_size = 128
        biLSTM_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_size))
        sentence_representation_biLSTM = biLSTM_layer(embeddings)

        # now, create some deep layers
        # deep 1
        dense1 = tf.keras.layers.Dense(256, activation='gelu')
        dropout1 = tf.keras.layers.Dropout(.2)
        output1 = dropout1(dense1(sentence_representation_biLSTM))

        # deep 2
        dense2 = tf.keras.layers.Dense(128, activation='gelu')
        dropout2 = tf.keras.layers.Dropout(.2)
        output2 = dropout2(dense2(output1))

        # deep 3
        dense3 = tf.keras.layers.Dense(64, activation='gelu')
        dropout3 = tf.keras.layers.Dropout(.2)
        output3 = dropout3(dense3(output2))

        # softmax
        softmax_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
        final_output = softmax_layer(output3)

        # combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])

        # compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy', # change to binary?
            metrics=[macro_cPrecision, macro_cRecall, macro_cF1,
                     micro_cPrecision, micro_cRecall, micro_cF1,
                     recall_c0, precision_c0, f1_c0,
                     recall_c1, precision_c1, f1_c1,
                     recall_c2, precision_c2, f1_c2,
                     recall_c3, precision_c3, f1_c3,
                     recall_c4, precision_c4, f1_c4
             ]
            # TODO - what metrics to report for multilabel? macro/micro F1, etc..?
        )
