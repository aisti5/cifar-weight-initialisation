import datetime
import os

import keras
import matplotlib.pylab as plt
import numpy as np
import sklearn.model_selection
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential

import utils.filesystem
from cifar_classifiers.BaseClassifier import BaseClassifier


class CNNClassifier(BaseClassifier):
    def __init__(self, **kwargs):
        super(CNNClassifier, self).__init__(kwargs.get('dataset', None))
        self.batch_size = kwargs.get('batch_size', 64)
        self.num_classes = self.dataset.cifar_size
        self.epochs = kwargs.get('epochs', 100)
        self.learning_rate = kwargs.get('lr', 0.001)
        self.data_augmentation = True

        self.seed = 0xDEADBEEF
        self.conv_init = kwargs.get('conv_init', keras.initializers.lecun_normal(self.seed))
        self.dense_init = kwargs.get('dense_init', keras.initializers.lecun_uniform(self.seed))
        self.bias_init = kwargs.get('bias_init', keras.initializers.Ones())

        self.model_name = "keras_cifar-{}".format(self.dataset.cifar_size)
        self.model = None  # type: keras.Model

        log_dir = os.path.join(self.save_dir, '_'.join(
            [self.model_name, 'b' + str(self.batch_size), 'e' + str(self.epochs),
             datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")]))

        if not kwargs.get('debug', False):
            self.logger.warning("Running in DEBUG mode: no output will be saved.")
            self.callbacks = [
                keras.callbacks.TensorBoard(
                    log_dir=log_dir, batch_size=self.batch_size, write_grads=True, write_images=False, histogram_freq=1),
                keras.callbacks.ModelCheckpoint(os.path.join(log_dir, self.model_name + "_e{epoch}.h5"),
                                                monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False,
                                                mode='auto', period=1)
            ]
        else:
            utils.filesystem.make_dir(log_dir)
            self.logger.info("Saving output of the run in: {}".format(log_dir))
            self.callbacks = []

        self._build_model(visulise_cifar_samples=kwargs.get('visualise_cifar_samples', False))

    def __str__(self):
        return 'CNNCfr'

    def __repr__(self):
        return str(self)

    def _build_model(self, visulise_cifar_samples=True):
        """
        Build the CNN.

        :param utils.CIFARDataset.CIFARDataset dataset:
        :return:
        """
        if visulise_cifar_samples:
            selection = np.random.choice(self.dataset.train_img_data.shape[0], 9)
            fig = plt.figure()
            for subplot_i, cifar_idx in enumerate(selection):
                plt.subplot(3, 3, subplot_i + 1)
                plt.imshow(self.dataset.train_img_data[cifar_idx])
            plt.show()

        self.logger.info("Building the CNN...")
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=(32, 32, 3),
                         kernel_initializer=self.conv_init,
                         bias_initializer=self.bias_init))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), kernel_initializer=self.conv_init, bias_initializer=self.bias_init))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(
            Conv2D(64, (3, 3), padding='same', kernel_initializer=self.conv_init, bias_initializer=self.bias_init))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), kernel_initializer=self.conv_init, bias_initializer=self.bias_init))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, kernel_initializer=self.dense_init, bias_initializer=self.bias_init))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.dataset.cifar_size, kernel_initializer=self.dense_init, bias_initializer=self.bias_init))
        model.add(Activation('softmax', name='softmax'))

        opt = keras.optimizers.adam(lr=self.learning_rate, decay=1e-6)
        # Let's train the model using RMSprop
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=opt,
                      metrics=[keras.metrics.mae, keras.metrics.categorical_accuracy])

        self.model = model

    def train(self):
        super(CNNClassifier, self).train()
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(self.dataset.train_img_data,
                                                                                    self.dataset.train_img_categories,
                                                                                    test_size=.2,
                                                                                    random_state=0xDEADBEEF)
        self.model.fit(x_train, y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_data=(x_test, y_test),
                       shuffle=True, callbacks=self.callbacks)

    def kfold_cv(self, n_folds):
        super(CNNClassifier, self).kfold_cv(n_folds)
        kfold = sklearn.model_selection.StratifiedKFold(n_splits=n_folds, random_state=0xDEADBEEF, shuffle=True)
        for train_idx, test_idx in kfold.split(
                self.dataset.train_img_data.reshape(self.dataset.train_img_data.shape[0], -1),
                np.argmax(self.dataset.train_img_categories, axis=1)):
            train_x = self.dataset.train_img_data[train_idx]
            train_y = self.dataset.train_img_categories[train_idx]
            test_x = self.dataset.test_img_data[test_idx]
            test_y = self.dataset.test_img_categories[test_idx]

            self.model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=self.epochs,
                           batch_size=self.batch_size, callbacks=self.callbacks)
