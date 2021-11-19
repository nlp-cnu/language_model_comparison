from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping
import tensorflow_addons as tfa
import time

from DataGenerator import *
from Metrics import *


class WriteMetrics(Callback):
    """
    Callback method to write metrics.
    Writes epoch, loss, macro_cPrecision, macro_cRecall, macro_cF1, micro_cPrecision, micro_cRecall, micro_cF1, time(s)
    """

    def __init__(self, metric_file=None):
        self.metric_file = metric_file
        self.start = 0

    def on_epoch_start(self, epoch, logs=None):
        self.start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        time_taken = int((time.time() - self.start) * 60)
        keys = list(logs.keys())
        print("End of epoch {}; log keys;: {}".format(epoch + 1, keys))
        print(list(logs.values()))
        vals = list(logs.values())
        with open(self.metric_file, 'a') as file:
            file.write(
                "{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{}\n".format(epoch + 1, vals[0], vals[1], vals[2]
                                                                               , vals[3], vals[4], vals[5],
                                                                               vals[6], vals[7], time_taken))


class Classifier:
    """
    Classifier class, which holds a language model and a classifier
    This class can be modified to create whatever architecture you want,
    however it requires the following instance variables:
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
    """
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
        """
        Initializer for a language model. This class should be extended, and
        the model should be built in the constructor. This constructor does
        nothing, since it is an abstract class. In the constructor however
        you must define:
        self.tokenizer
        self.model
        """
        self.tokenizer = None
        self.model = None
        self.metric_file = None

    def train(self, x, y, batch_size=BATCH_SIZE, validation_data=None, epochs=EPOCHS):
        """
        Trains the classifier
        :param x: the training data
        :param y: the training labels
        :param batch_size: the batch size
        :param validation_data: a tuple containing x and y for a validation dataset so, validation_data[0] = val_x and
        validation_data[1] = val_y
        :param epochs: the number of epochs to train for
        """

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
            callbacks=[WriteMetrics(self.metric_file), EarlyStopping(monitor='loss', min_delta=0.0001, patience=5,
                                                                     mode='min', restore_best_weights=True)]
        )

    # function to predict using the NN
    def predict(self, x, batch_size=BATCH_SIZE):
        """
        Predicts labels for data
        :param x: data
        :param batch_size: size of batch, constant defined earlier
        :return: predictions
        """
        if not isinstance(x, tf.keras.utils.Sequence):
            # TODO - max length can be longer depending on the model -e.g. Roberta is 768
            tokenized = self.tokenizer(x, padding=True, truncation=True, max_length=512, return_tensors='tf')
            x = (tokenized['input_ids'], tokenized['attention_mask'])

        return self.model.predict(x, batch_size=batch_size)


# NEEDS TO BE UPDATED - ONLY MULTI WORKS RIGHT NOW
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
        dropout1 = tf.keras.layers.Dropout(.8)
        output1 = dropout1(dense1(sentence_representation_biLSTM))
        # output1 = dropout1(dense1(sentence_representation_language_model))

        # deep 2
        dense2 = tf.keras.layers.Dense(128, activation='gelu')
        dropout2 = tf.keras.layers.Dropout(.8)
        output2 = dropout2(dense2(output1))

        # deep 3
        dense3 = tf.keras.layers.Dense(64, activation='gelu')
        dropout3 = tf.keras.layers.Dropout(.8)
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
        )


class MultiLabel_Text_Classifier(Classifier):
    def __init__(self, language_model_name, num_classes, rate, metric_file, language_model_trainable=False):
        """
        This is identical to the Binary_Text_Classifier, except the last layer uses
        a softmax, loss is Categorical Cross Entropy and its output dimension is num_classes
        Also, different metrics are reported.
        """
        Classifier.__init__(self)
        self.language_model_name = language_model_name
        self.num_classes = num_classes
        self.rate = rate
        self.metric_file = metric_file

        # create the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

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
        dropout1 = tf.keras.layers.Dropout(.8)
        output1 = dropout1(dense1(sentence_representation_biLSTM))

        # deep 2
        dense2 = tf.keras.layers.Dense(128, activation='gelu')
        dropout2 = tf.keras.layers.Dropout(.8)
        output2 = dropout2(dense2(output1))

        # deep 3
        dense3 = tf.keras.layers.Dense(64, activation='gelu')
        dropout3 = tf.keras.layers.Dropout(.8)
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
            loss='categorical_crossentropy',
            metrics=[macro_cPrecision, macro_cRecall, macro_cF1,
                     micro_cPrecision, micro_cRecall, micro_cF1,
                     recall_c0, precision_c0, f1_c0,
                     recall_c1, precision_c1, f1_c1,
                     recall_c2, precision_c2, f1_c2,
                     recall_c3, precision_c3, f1_c3,
                     recall_c4, precision_c4, f1_c4
                     ]
        )
