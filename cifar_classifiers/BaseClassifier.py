import abc
import os

import utils.filesystem
from utils.FormattedLogger import FormattedLogger

if False:
    from utils.CIFARDataset import CIFARDataset


class BaseClassifier(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, dataset):
        self.logger = FormattedLogger(str(self))
        self.save_dir = os.path.join('models', 'snapshots', str(self))
        self.dataset = dataset  # type: CIFARDataset
        utils.filesystem.make_dir(self.save_dir)

    @staticmethod
    def _remove_wrapper_args(**kwargs):
        for arg in ['dataset', 'debug']:
            kwargs.pop(arg)
        return kwargs

    @abc.abstractmethod
    def train(self):
        """
        Train the classifier on x to predict y.
        """
        self.logger.info("Running a train-test split with {}.".format(str(self)))

    @abc.abstractmethod
    def classify(self, x):
        """
        Classify x.

        :param np.array x:  data
        :return:            predicted class
        :rtype:             np.array
        """

    def kfold_cv(self, n_folds):
        self.logger.info("Running {}-fold cross-validation with {}.".format(n_folds, str(self)))
