from tensorflow.keras.callbacks import Callback, EarlyStopping
import numpy as np

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
class NoChange(EarlyStopping):
    losses = []

    def on_epoch_end(self, epoch, logs=None):
        if len(self.losses) < 5:
            self.losses.append(logs["loss"])
        else:
            if self.losses[0] == self.losses[-1]:
                self.model.stop_training = True


# callback = EarlyStopping(monitor='loss', patience=5)
# stops training if loss has no improvement for 5  consecutive epochs... interesting

class EarlyStoppingAtMinLoss(Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.

  taken from: https://www.tensorflow.org/guide/keras/custom_callback#early_stopping_at_minimum_loss
  """

    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


