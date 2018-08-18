from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

class ScorerFactory:
    def get_scorer_from_configuration(self, configuration):
        return None

class Scorer:
    def get_score(self, expected, predicted, classes):
        pass

class AccuracyScorer(Scorer):
    def get_score(self, expected, predicted, classes):
        return accuracy_score(expected, predicted)

class RecallScorer(Scorer):
    def get_score(self, expected, predicted, classes):
        return recall_score(expected, predicted, average='weighted', labels=classes)

class PrecisionScore(Scorer):
    def get_score(self, expected, predicted, classes):
        return precision_score(expected, predicted, average='weighted', labels=classes)

class F1Scorer(Scorer):
    def get_score(self, expected, predicted, classes):
        return f1_score(expected, predicted, average='weighted', labels=classes)

