import os
import sys

import keras.utils
import numpy as np

from preprocessing.normalisation import min_max_norm
from utils.FormattedLogger import FormattedLogger

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class CIFARDatasetException(Exception):
    pass


class CIFARDataset(object):
    """Container class for CIFAR train/test data."""
    def __init__(self, cifar_size):
        """
        Initialise datset container.

        :param int cifar_size:      Type of CIFAR dataset: either 10 for CIFAR-10 or 100 for CIFAR-100.
        """
        # Verify input
        if cifar_size in [10, 100]:
            self.cifar_size = cifar_size
        else:
            raise CIFARDatasetException("CIFAR dataset type has to be 10 or 100, got {}".format(cifar_size))
        self.cifar_size = cifar_size
        self.dataset_path = None
        self.logger = FormattedLogger('CIFARDataset')

        self.category_labels = None
        self.train_img_data = None
        self.train_img_categories = None

        self.test_img_data = None
        self.test_img_categories = None

    def load(self, dataset_path, truncate_to=None):
        """
        Load dataset from a local directory.

        :param str dataset_path:    Path to a directory to load the dataset from.
        :param int truncate_to:     If specified, only the first truncate_to samples will be loaded.
        :raises:                    CIFARDatasetException
        """
        def _unpickle(file_handle):
            """Python 2/3 compatible uncpikling of CIFAR data."""
            if sys.version_info[0] == 2:
                return pickle.load(file_handle)
            else:
                return pickle.load(file_handle, encoding='latin1')

        self.logger.info("Loading CIFAR-{} data from {}".format(self.cifar_size, dataset_path))
        if os.path.exists(dataset_path):
            self.dataset_path = dataset_path
        else:
            raise CIFARDatasetException("Path {} does not exist.".format(dataset_path))

        train_img_data = np.array([])
        train_img_categories = np.array([])
        try:
            if self.cifar_size == 10:
                meta_path = os.path.join(dataset_path, 'batches.meta')
                with open(meta_path, 'rb') as meta_handle:
                    meta_dict = _unpickle(meta_handle)
                self.category_labels = meta_dict['label_names']

                self.logger.info("Loading test data...")
                test_data_path = os.path.join(dataset_path, 'test_batch')
                with open(test_data_path, 'rb') as test_data_handle:
                    test_dict = _unpickle(test_data_handle)
                self.test_img_categories = np.array(test_dict['labels'])
                self.test_img_data = test_dict['data']

                self.logger.info("Loading training data...")
                batch_idx = range(1, 6)
                for idx in batch_idx:
                    batch_path = os.path.join(dataset_path, 'data_batch_{}'.format(idx))
                    with open(batch_path, 'rb') as batch_handle:
                        batch_dict = _unpickle(batch_handle)
                        train_img_data = batch_dict['data'] if train_img_data.size == 0 else np.vstack(
                            (train_img_data, batch_dict['data']))
                        train_img_categories = np.append(train_img_categories, batch_dict['labels'])
                    self.logger.debug("Loaded batch {}".format(batch_path))
            elif self.cifar_size == 100:
                # ::todo: implement CIFAR-100 loading
                raise NotImplementedError("CIFAR-100 loading not yet implemented.")
        except (OSError, IOError, FileNotFoundError) as ex:
            self.logger.error("Could not load the dataset from {}".format(dataset_path))
            raise CIFARDatasetException("{}".format(ex))

        if truncate_to is not None:
            self.logger.info("Loading only the first {} images of the training set.".format(truncate_to))
            train_img_categories = train_img_categories[:truncate_to]
            train_img_data = train_img_data[:truncate_to]

        self.train_img_data = train_img_data
        self.train_img_categories = train_img_categories
        self.logger.info("\nCIFAR-{cifar_type} Summary:\nLoaded categories: {cat_labels}\n"
                         "Loaded {n_train} instances of training samples\n"
                         "Loaded {n_test} instances of test samples\n".format(
            cifar_type=self.cifar_size,
            cat_labels=', '.join(self.category_labels),
            n_train=self.train_img_categories.size,
            n_test=self.test_img_categories.size))

    def pre_process_for_cnn(self, norm_func=min_max_norm, **kwargs):
        """
        Apply pre-processing functions for CNN processing.

        :param func norm_func:      Function used to normalise the inputs.
        :param kwargs:              kwargs are passed on to norm_func.
        :return:                    Return a copy of the dataset.
        :rtype:                     CIFARDataset
        """
        self.logger.info("Reshaping CIFAR images...")
        try:
            self.train_img_data = norm_func(
                self.train_img_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), **kwargs)
            self.train_img_categories = keras.utils.to_categorical(self.train_img_categories, self.cifar_size)
            self.test_img_data = norm_func(
                self.train_img_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), **kwargs)
            self.test_img_categories = keras.utils.to_categorical(self.test_img_categories, self.cifar_size)
        except TypeError as ex:
            raise CIFARDatasetException("Error pre-processing data: {}".format(ex))


if __name__ == '__main__':
    data_path = os.path.join('data', 'cifar-10-batches-py')
    cifar10 = CIFARDataset(10)
    cifar10.load(data_path)
