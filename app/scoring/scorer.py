from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import numpy as np
import tensorflow as tf
from keras import backend as keras_backend

class ScorerFactory:
    def get_scorers(self, evolution_configuration):
        scoring_configurations = evolution_configuration.scores

        return [self._get_scorer_from_configuration(configuration) for configuration in scoring_configurations]

    def _get_scorer_from_configuration(self, configuration):
        return {
            'accuracy': AccuracyScorer(),
            'recall': RecallScorer(),
            'precision': PrecisionScorer(),
            'f1': F1Scorer()
        }.get(configuration["name"], AccuracyScorer())

class Scorer:
    def get_score(self, expected, predicted, classes):
        pass

class AccuracyScorer(Scorer):
    def get_score(self, expected, predicted, classes):
        return accuracy_score(expected, predicted)

class RecallScorer(Scorer):
    def get_score(self, expected, predicted, classes):
        return recall_score(expected, predicted, average='weighted', labels=classes)

class PrecisionScorer(Scorer):
    def get_score(self, expected, predicted, classes):
        return precision_score(expected, predicted, average='weighted', labels=classes)

class F1Scorer(Scorer):
    def get_score(self, expected, predicted, classes):
        return f1_score(expected, predicted, average='weighted', labels=classes)

class MeanIntersectionOverUnion():
    def get_score(self, y_true, y_pred):
        score = self._calculate_score(y_true, y_pred)

        return score

    def _calculate_score(self, y_true, y_pred):
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            y_pred_ = tf.to_int32(y_pred > t)
            score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
            keras_backend.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([up_opt]):
                score = tf.identity(score)
            prec.append(score)
        return keras_backend.mean(keras_backend.stack(prec), axis=0)
    