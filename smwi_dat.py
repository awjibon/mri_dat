import gc

import tensorflow as tf
import numpy as np

class Constants:
    def __init__(self):
        self.gt_max = 6.84
        self.symmeteric_model= 'net_vgg_sym'
        self.paired_model = 'net_vgg_paired'
        self.right_model = 'net_vgg_right'
        self.left_model = 'net_vgg_left'
        self.model_type_symmetric = 'symmetric'
        self.model_type_paired = 'paired'
        self.model_type_separate = 'separate'
        self.model_types = {self.model_type_symmetric: self.symmeteric_model, self.model_type_paired: self.paired_model,
                            self.model_type_separate:[self.right_model, self.left_model]}


constants = Constants()


class SMWI_DAT:
    def __init__(self, model='symmetric'):
        self.model = model

    def predict_sbr(self, right_patch, left_patch):
        right_patch = np.reshape(right_patch, [1, 50, 50, 20, 1])
        left_patch = np.reshape(left_patch, [1, 50, 50, 20, 1])
        right_patch /= np.mean(right_patch)
        left_patch /= np.mean(left_patch)
        right_patch = np.flip(right_patch, axis=2)

        predictor, left_sbr, right_sbr = None, None, None
        if self.model == 'separate':
            predictor = tf.keras.models.load_model(constants.right_model, compile=False)
            right_sbr = predictor.predict(right_patch)
            del predictor
            gc.collect()
            predictor = tf.keras.models.load_model(constants.left_model, compile=False)
            left_sbr = predictor.predict(left_patch)
        else:
            predictor = tf.keras.models.load_model(constants.model_types[self.model], compile=False)
            left_sbr, right_sbr = predictor.predict([left_patch, right_patch, np.array([1.0]), np.array([1.0])])
        return right_sbr * constants.gt_max, left_sbr * constants.gt_max
