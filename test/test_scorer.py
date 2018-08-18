from unittest import TestCase
from app.scorer import AccuracyScorer, RecallScorer, F1Scorer, PrecisionScorer, ScorerFactory

class TestScorer(TestCase):
    def test_given_expected_and_predicted_labels_when_get_score_from_accuracy_scorer_then_the_accuracy_should_be_returned(self):
        expected = [1, 0, 1, 1, 0]
        predicted = [1, 0, 0, 0, 0]
        classes = ["Android", "Vulkan", "Human"]

        score = AccuracyScorer().get_score(expected, predicted, classes)

        self.assertEqual(0.6, score)

    def test_given_expected_and_predicted_labels_when_get_score_from_recall_scorer_then_the_recall_should_be_returned(self):
        expected = ["Android", "Vulkan", "Android", "Android", "Android", "Vulkan"]
        predicted = ["Android", "Vulkan", "Vulkan", "Vulkan", "Vulkan", "Vulkan"]
        classes = ["Android", "Vulkan"]

        score = RecallScorer().get_score(expected, predicted, classes)

        self.assertEqual(0.5, score)

    def test_given_expected_and_predicted_labels_when_get_score_from_precision_scorer_then_the_precision_should_be_returned(self):
        expected = ["Android", "Vulkan", "Android", "Android", "Android"]
        predicted = ["Android", "Vulkan", "Vulkan", "Vulkan", "Vulkan"]
        classes = ["Android", "Vulkan"]

        score = PrecisionScorer().get_score(expected, predicted, classes)

        self.assertEqual(0.85, score)

    def test_given_expected_and_predicted_labels_when_get_score_from_f1_scorer_then_the_f1_score_should_be_returned(self):
        expected = ["Android", "Vulkan", "Android", "Android", "Android"]
        predicted = ["Android", "Vulkan", "Vulkan", "Vulkan", "Vulkan"]
        classes = ["Android", "Vulkan"]

        score = F1Scorer().get_score(expected, predicted, classes)

        self.assertEqual(0.4, score)

    def test_given_a_configuration_when_get_scorer_then_the_correct_scorer_should_be_returned(self):
        factory = ScorerFactory()

        accuracy = factory.get_scorer_from_configuration({ "name": "accuracy", "minimize": False})
        recall = factory.get_scorer_from_configuration({ "name": "recall", "minimize": False})
        precision = factory.get_scorer_from_configuration({ "name": "precision", "minimize": False})
        f1_score = factory.get_scorer_from_configuration({ "name": "f1", "minimize": False})

        self.assertEqual(AccuracyScorer, type(accuracy))
        self.assertEqual(RecallScorer, type(recall))
        self.assertEqual(PrecisionScorer, type(precision))
        self.assertEqual(F1Scorer, type(f1_score))