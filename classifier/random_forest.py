import sys
sys.path.append('../')

from sklearn.svm import LinearSVC, SVC
from classifier.cross_validation import KFoldCrossVal
from utility.file_utility import FileUtility
from sklearn.ensemble import RandomForestClassifier


class RFClassifier:
    def __init__(self, X, Y):
        self.model = RandomForestClassifier(bootstrap=True, criterion='gini',
            min_samples_split= 2, max_features='auto', min_samples_leaf=1, n_estimators=1000)
        self.X = X
        self.Y = Y

    def tune_and_eval(self, results_file,
                      params=[{"n_estimators": [100, 200, 500, 1000],
              "criterion": [ "entropy"], #"gini",
              'max_features': ['sqrt'], # 'auto',
             'min_samples_split':[5], # 2,5,10
             'min_samples_leaf':[1]}]):#1,2,5
        self.CV = KFoldCrossVal(self.X, self.Y, folds=10)
        self.CV.tune_and_evaluate(self.model, parameters=params, score='f1_macro', file_name=results_file + '_RF' , n_jobs=15)
