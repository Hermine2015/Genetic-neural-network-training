from keras.preprocessing import image as img
from skimage import transform
import os
import numpy as np


class ObjectRecognitionImagePreparer:
    def get_resized_training_data(self, training_path, label_path, dimensions):

        train_ids = self._get_all_files_in_path(training_path)

        training_set = np.zeros((len(train_ids), dimensions[0], dimensions[1], dimensions[2]), dtype=np.uint8)
        training_label = np.zeros((len(train_ids), dimensions[0], dimensions[1], dimensions[2]), dtype=np.bool)

        for index, image_id in enumerate(train_ids):
            image = img.load_img(training_path + '/' + image_id)
            image_as_matrix = img.img_to_array(image)[:, :, 1]
            image_as_matrix = transform.resize(image_as_matrix, dimensions, mode='constant', preserve_range=True)

            training_set[index] = image_as_matrix

            mask_image = img.load_img(label_path + '/' +  image_id)
            mask = img.img_to_array(mask_image)[:, :, 1]
            training_label[index] = transform.resize(mask, dimensions, mode='constant', preserve_range=True)

        return training_set, training_label

    def _get_all_files_in_path(self, training_path):
        return next(os.walk(training_path))[2]