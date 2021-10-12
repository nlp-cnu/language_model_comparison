import tensorflow as tf
import numpy as np

#Class to generate batches
# The datagenerator inherits from the sequence class which is used to generate
# data for each batch of training. Using a sequence generator is much more
# in terms of training time because it allows for variable size batches. (depending
# on the maximum length of sequences in the batch)
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, tokenizer, shuffle=True):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tokenizer = tokenizer

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        #TODO - max length may be > 512 depending on the classifier
        tokenized = self.tokenizer(batch_x, padding=True, truncation=True, max_length=512, return_tensors='tf')
                                                     
        return (tokenized['input_ids'], tokenized['attention_mask']), batch_y
    
    def on_epoch_end(self):
        """
        Method is called each time an epoch ends. This will shuffle the data at
        the end of an epoch, which ensures the batches are not identical each 
        epoch (therefore improving performance)
        :return:
        """
        if self.shuffle:
            idxs = np.arange(len(self.x))
            np.random.shuffle(idxs)
            self.x = [self.x[idx] for idx in idxs]
            self.y = self.y[idxs]
