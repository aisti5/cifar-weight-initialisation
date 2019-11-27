import sklearn.model_selection
import xgboost as xgb

from cifar_classifiers.BaseClassifier import BaseClassifier


class UnifiedXGBClassifier(BaseClassifier):
    def __init__(self, **kwargs):
        super(UnifiedXGBClassifier, self).__init__(kwargs.get('dataset', None))
        self.eval_metric = kwargs.pop('eval_metric', 'mlogloss')
        self.lr = kwargs.pop('lr', 0.1)
        kwargs = super(UnifiedXGBClassifier, self)._remove_wrapper_args(**kwargs)
        self.classifier = xgb.XGBClassifier(**kwargs, silent=True, n_jobs=-1)

    def __str__(self):
        return 'XGBCfr'

    def __repr__(self):
        return str(self)

    def train(self):
        super(UnifiedXGBClassifier, self).train()
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(self.dataset.train_img_data,
                                                                                    self.dataset.train_img_categories,
                                                                                    test_size=.2,
                                                                                    random_state=0xDEADBEEF)
        self.classifier.fit(x_train, y_train, eval_metric=self.eval_metric, verbose=True)
        test_prediction = self.classify(x_test)
        test_acc = sklearn.metrics.classification.accuracy_score(test_prediction, y_test)
        self.logger.info("Accuracy: {0:.2f})".format(test_acc))

    def classify(self, x):
        return self.classifier.predict(x)

    def kfold_cv(self, n_folds):
        super(UnifiedXGBClassifier, self).kfold_cv(n_folds)
        k_fold = sklearn.model_selection.KFold(n_splits=n_folds, random_state=0xDEADBEEF, shuffle=True)
        results = sklearn.model_selection.cross_val_score(self.classifier, self.dataset.train_img_data,
                                                          self.dataset.train_img_categories, cv=k_fold,
                                                          verbose=3)
        self.logger.info("Accuracy: {0:.2f} (std={0:.3f})".format(results.mean(), results.std()))
        return results

    def visualise(self):
        xgb.plot_tree(self.classifier)
