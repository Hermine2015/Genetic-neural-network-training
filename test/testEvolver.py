from unittest import TestCase
from app.evolver import Evolver
from unittest.mock import patch

class TestEvolver(TestCase):

    @patch('app.data_preparation.image_preparer.ObjectRecognitionImagePreparer.get_resized_training_data')
    def test_given_an_odd_number_if_rounded_when_make_even_then_one_should_be_added_to_the_number(self, preparation_mock):
        preparation_mock.return_value = [], []

        result = Evolver(None, None, None)._make_even(3.421)

        self.assertEqual(4, result)

    @patch('app.data_preparation.image_preparer.ObjectRecognitionImagePreparer.get_resized_training_data')
    def test_given_an_even_number_if_rounded_when_make_even_then_the_same_number_as_an_integer_should_be_returned(self, preparation_mock):
        preparation_mock.return_value = [], []

        result = Evolver(None, None, None)._make_even(2.421)

        self.assertEqual(2, result)

    @patch('app.data_preparation.image_preparer.ObjectRecognitionImagePreparer.get_resized_training_data')
    def test_given_a_float_closer_to_the_value_below_when_get_int_then_the_float_should_be_rounded_to_the_int_below(self, preparation_mock):
        preparation_mock.return_value = [], []

        result = Evolver(None, None, None)._get_int(2.345)

        self.assertEqual(2, result)

    @patch('app.data_preparation.image_preparer.ObjectRecognitionImagePreparer.get_resized_training_data')
    def test_given_a_float_closer_to_the_value_above_when_get_int_then_the_float_should_be_rounded_to_the_int_above(self, preparation_mock):
        preparation_mock.return_value = [], []

        result = Evolver(None, None, None)._get_int(2.765)

        self.assertEqual(3, result)

    @patch('app.data_preparation.image_preparer.ObjectRecognitionImagePreparer.get_resized_training_data')
    def test_given_total_and_a_value_when_get_tuple_then_a_tuple_of_the_size_of_the_total_with_repeating_values_should_be_returned(
            self, preparation_mock):
        preparation_mock.return_value = [], []

        result = Evolver(None, None, None)._get_tuple(2, 4)

        self.assertEqual((4, 4), result)
