from unittest import TestCase
from unittest.mock import patch
import numpy as np
from unittest.mock import call
from app.data_preparation.image_preparer import ObjectRecognitionImagePreparer

class TestImagePreparer(TestCase):

    @patch('keras.preprocessing.image.load_img')
    @patch('app.data_preparation.image_preparer.ObjectRecognitionImagePreparer._get_all_files_in_path')
    @patch('skimage.transform.resize')
    @patch('keras.preprocessing.image.img_to_array')
    def test_given_a_path_and_dimensions_when_get_resized_training_data_then_the_images_should_be_returned_as_a_set_of_matrices(self,
                         to_array_mock, resize_mock, list_mock, load_mock):

        as_matrix = np.array([[[131, 131, 131],
                       [95, 95, 95]],
                      [[132, 132, 132],
                       [85, 85, 85]]])

        resized_1 = np.array([[[131, 131],
                               [95, 95]],
                              [[132, 132],
                               [85, 85]]])

        resized_2 = np.array([[[12, 12],
                               [131, 131]],
                              [[85, 85],
                               [0, 0]]])

        list_mock.return_value = ['first.jpg', 'second.jpg']
        to_array_mock.return_value = as_matrix
        resize_mock.side_effect = [resized_1, resized_2, resized_2, resized_1]

        dimensions = (2, 2, 2)

        training_data, training_labels = ObjectRecognitionImagePreparer().get_resized_training_data('train_path', 'label_path', dimensions)

        load_mock.assert_has_calls([
            call('train_path/first.jpg'),
            call('label_path/first.jpg'),
            call('train_path/second.jpg'),
            call('label_path/second.jpg')
        ])

        arguments, kwargs = resize_mock.call_args_list[0]
        self.assertListEqual([[131,  95], [132,  85]], arguments[0].tolist())
        self.assertListEqual(training_data.tolist(), [resized_1.tolist(), resized_2.tolist()])
        expected_labels = [[[[True, True], [True, True]], [[True, True], [False, False]]],
                           [[[True, True], [True, True]], [[True, True], [True, True]]]]
        self.assertListEqual(training_labels.tolist(), expected_labels)