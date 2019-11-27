import os
import copy

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.stflow import LogFileWriter
from sacred.utils import apply_backspaces_and_linefeeds

try:
    from cifar_classifiers.UnifiedXGBClassifier import UnifiedXGBClassifier
except ImportError:
    import sys

    project_path = os.path.abspath(os.getcwd())
    print("Appending {} to sys.path".format(project_path))
    sys.path.append(project_path)
    from cifar_classifiers.UnifiedXGBClassifier import UnifiedXGBClassifier

from cifar_classifiers.CNNClassifier import CNNClassifier
from utils.CIFARDownloader import CIFARDownloader
from utils.CIFARDataset import CIFARDataset

if False:
    from cifar_classifiers.BaseClassifier import BaseClassifier

experiment = Experiment('cifar-weights')
experiment.captured_out_filter = apply_backspaces_and_linefeeds
experiment.observers.append(MongoObserver.create())


@experiment.config
def config():
    experiment_config = dict()
    debug = False

    n_samples = 100
    cifar_size = 10
    n_folds = 5
    experiment_config['meta'] = dict({'cifar_size': cifar_size, 'n_folds': n_folds, 'n_samples': n_samples})

    epochs = 20
    batch_size = 64
    lr = 0.001
    xgb_max_depth = 6
    experiment_config['classifiers'] = [{'class': CNNClassifier,
                                         'params': {'epochs': epochs, 'batch_size': batch_size, 'lr': lr}},
                                        {'class': UnifiedXGBClassifier,
                                         'params': {'max_depth': xgb_max_depth}, 'lr': 0.1, 'eval_metric': 'mlogloss'}]


@experiment.automain
@LogFileWriter(experiment)
def main(debug, experiment_config):
    cifar_size = experiment_config['meta']['cifar_size']
    n_samples = experiment_config['meta']['n_samples']
    n_folds = experiment_config['meta']['n_folds']

    cifar10_downloader = CIFARDownloader(cifar_size=cifar_size)
    cifar10_dataset = CIFARDataset(cifar_size=cifar_size)
    cifar10_dataset.load(cifar10_downloader.get_data(), truncate_to=n_samples)
    cifar10_dataset_cnn = copy.copy(cifar10_dataset)
    cifar10_dataset_cnn.pre_process_for_cnn(x_min=0, x_max=1)

    for classifier_config in experiment_config['classifiers']:
        dataset = cifar10_dataset_cnn if 'CNN' in str(classifier_config['class']) else cifar10_dataset
        params = classifier_config['params']  # type: dict
        # params['dataset'] = dataset
        # params['debug'] = debug
        classifier = classifier_config['class'](**params)  # type: BaseClassifier
        if n_folds > 0:
            classifier.kfold_cv(n_folds)
        else:
            classifier.train()
