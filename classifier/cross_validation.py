__author__ = "Ehsaneddin Asgari"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu or ehsaneddin.asgari@helmholtz-hzi.de"
__project__ = "LLP - MicroPheno"
__website__ = "https://llp.berkeley.edu/micropheno/"

import sys
sys.path.append('../')
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from utility.file_utility import FileUtility
from sklearn.metrics import f1_score,confusion_matrix


class CrossValidator(object):
    '''
     The Abstract Cross-Validator
    '''
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.scoring =  {'precision_micro': 'precision_micro', 'precision_macro': 'precision_macro', 'recall_macro': 'recall_macro','recall_micro': 'recall_micro', 'f1_macro':'f1_macro', 'f1_micro':'f1_micro'}


class KFoldCrossVal(CrossValidator):
    '''
        K-fold cross-validation tuning and evaluation
    '''
    def __init__(self, X, Y, folds=10, random_state=1):
        '''
        :param X:
        :param Y:
        :param folds:
        :param random_state:
        '''
        CrossValidator.__init__(self, X, Y)
        self.cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        self.X = X
        self.Y = Y

    def tune_and_evaluate(self, estimator, parameters, score='macro_f1', n_jobs=-1, file_name='results'):
        '''
        :param estimator:
        :param parameters:p
        :param score:
        :param n_jobs:
        :param file_name: directory/tuning/classifier/features/
        :return:
        '''
        # greed_search
        self.greed_search = GridSearchCV(estimator=estimator, param_grid=parameters, cv=self.cv, scoring=self.scoring,
                                         refit=score, error_score=0, n_jobs=n_jobs)

        label_set=list(set(self.Y))
        # fitting
        self.greed_search.fit(X=self.X, y=self.Y)
        y_predicted = cross_val_predict(self.greed_search.best_estimator_, self.X, self.Y)
        conf=confusion_matrix(self.Y,y_predicted,labels=label_set)
        # save in file
        FileUtility.save_obj(file_name, [label_set, conf, self.greed_search.best_score_, self.greed_search.best_estimator_, self.greed_search.cv_results_, self.greed_search.best_params_, y_predicted])


