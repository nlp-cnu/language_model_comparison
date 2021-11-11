'''
Examples of custom metrics
'''

from keras import backend as K
import tensorflow as tf


### Right now, I hardcode each of the metrics per class number
### And for the number of classes, however there should be a better
### way to do this with objects (instaniate with num_classes, and class_num)
### Depending on the function that you are computing
### So, right now, register one of these to actually call. Then the computation
### Is done using the non-custom (no c) functions
def recall_c0(y_true, y_pred):
    return class_recall(y_true, y_pred, 0)


def recall_c1(y_true, y_pred):
    return class_recall(y_true, y_pred, 1)


def recall_c2(y_true, y_pred):
    return class_recall(y_true, y_pred, 2)


def recall_c3(y_true, y_pred):
    return class_recall(y_true, y_pred, 3)


def recall_c4(y_true, y_pred):
    return class_recall(y_true, y_pred, 4)


def precision_c0(y_true, y_pred):
    return class_precision(y_true, y_pred, 0)


def precision_c1(y_true, y_pred):
    return class_precision(y_true, y_pred, 1)


def precision_c2(y_true, y_pred):
    return class_precision(y_true, y_pred, 2)


def precision_c3(y_true, y_pred):
    return class_precision(y_true, y_pred, 3)


def precision_c4(y_true, y_pred):
    return class_precision(y_true, y_pred, 4)


def f1_c0(y_true, y_pred):
    return class_f1(y_true, y_pred, 0)


def f1_c1(y_true, y_pred):
    return class_f1(y_true, y_pred, 1)


def f1_c2(y_true, y_pred):
    return class_f1(y_true, y_pred, 2)


def f1_c3(y_true, y_pred):
    return class_f1(y_true, y_pred, 3)


def f1_c4(y_true, y_pred):
    return class_f1(y_true, y_pred, 4)


def macro_cF1(y_true, y_pred):
    return macro_f1(y_true, y_pred, 5)


def macro_cRecall(y_true, y_pred):
    return macro_recall(y_true, y_pred, 5)


def macro_cPrecision(y_true, y_pred):
    return macro_precision(y_true, y_pred, 5)


def micro_cF1(y_true, y_pred):
    return micro_f1(y_true, y_pred, 5)


def micro_cRecall(y_true, y_pred):
    return micro_recall(y_true, y_pred, 5)


def micro_cPrecision(y_true, y_pred):
    return micro_precision(y_true, y_pred, 5)


# Macro-Averaged Prec, Recall, F1
def macro_f1(y_true, y_pred, num_classes):
    sum = 0
    for i in range(num_classes):
        sum += class_f1(y_true, y_pred, i)
    return sum / num_classes


def macro_precision(y_true, y_pred, num_classes):
    sum = 0
    for i in range(num_classes):
        sum += class_precision(y_true, y_pred, i)
    return sum / num_classes


def macro_recall(y_true, y_pred, num_classes):
    sum = 0
    for i in range(num_classes):
        sum += class_recall(y_true, y_pred, i)
    return sum / num_classes


# Micro-Averaged Prec, Recall, F1
def micro_precision(y_true, y_pred, num_classes):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def micro_recall(y_true, y_pred, num_classes):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def micro_f1(y_true, y_pred, num_classes):
    precision = micro_precision(y_true, y_pred, num_classes)
    recall = micro_recall(y_true, y_pred, num_classes)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# Class-specific Prec, Recall, F1
def class_recall(y_true, y_pred, class_num):
    class_y_true = tf.gather(y_true, [class_num], axis=1)
    class_y_pred = tf.gather(y_pred, [class_num], axis=1)
    true_positives = K.sum(K.round(K.clip(class_y_true * class_y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(class_y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def class_precision(y_true, y_pred, class_num):
    class_y_true = tf.gather(y_true, [class_num], axis=1)
    class_y_pred = tf.gather(y_pred, [class_num], axis=1)
    true_positives = K.sum(K.round(K.clip(class_y_true * class_y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(class_y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def class_f1(y_true, y_pred, class_num):
    precision = class_precision(y_true, y_pred, class_num)
    recall = class_recall(y_true, y_pred, class_num)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# Method to test, prints actual values so that you can debug
# Use a small enough batch size (below 10) so that you can print
# all the values to screen
def test(y_true, y_pred):
    class_num = 2
    num_classes = 5

    class_y_true = tf.gather(y_true, [class_num], axis=1)
    class_y_pred = tf.gather(y_pred, [class_num], axis=1)
    true_positives = K.sum(K.round(K.clip(class_y_true * class_y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(class_y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(class_y_pred, 0, 1)))
    y_pred_rounded = K.round(K.clip(y_pred, 0, 1))

    tf.print("y_true = ")
    tf.print(y_true)
    tf.print("y_pred = ")
    tf.print(y_pred)
    tf.print("y_pred_rounded = ")
    tf.print(y_pred_rounded)
    tf.print("class_y_true = ")
    tf.print(class_y_true)
    tf.print("class_y_pred = ")
    tf.print(class_y_pred)
    tf.print("true_positives = ")
    tf.print(true_positives)
    tf.print("possible_positives = ")
    tf.print(possible_positives)
    tf.print("predicted_positives = ")
    tf.print(predicted_positives)

    tf.print("macro_f1 = ")
    tf.print(macro_f1(y_true, y_pred, num_classes))
    tf.print("micro_f1 = ")
    tf.print(micro_f1(y_true, y_pred, num_classes))
    tf.print("class " + str(class_num) + " f1 = ")
    tf.print(class_f1(y_true, y_pred, class_num))
    tf.print("\n")

    return 1

# Note: These custom methods have been fully tested for text classification tasks
#      The results are correct. They conflict with the built in methods below:
#      Do not use the build in methods, use your own and verify they work by
#      actually looking at output and calculating scores
# tfa.metrics.F1Score(self._num_classes, average='micro', name='micro_f1'),
# tfa.metrics.F1Score(self._num_classes, average='macro', name='macro_f1'),
# tf.keras.metrics.Precision(class_id=0),
# tf.keras.metrics.Recall(class_id=0),
# tf.keras.metrics.Precision(class_id=1),
# tf.keras.metrics.Recall(class_id=1),
# tf.keras.metrics.Precision(class_id=2),
# tf.keras.metrics.Recall(class_id=2),
# tf.keras.metrics.Precision(class_id=3),
# tf.keras.metrics.Recall(class_id=3),
# tf.keras.metrics.Precision(class_id=4),
# tf.keras.metrics.Recall(class_id=4),