'''Traing'''
import utils
import svmutil
from grid import find_parameters

TRAIN_DATA = 'data/training_data_libsvm'
TEST_DATA = 'data/testing_data_libsvm'
BARE_DATA = 'data/training.data'
MODEL_PATH = 'model/speech.model'

training_data, testing_data = utils.load_data(BARE_DATA, 20000)

utils.format_data(training_data, TRAIN_DATA)
utils.format_data(testing_data, TEST_DATA)