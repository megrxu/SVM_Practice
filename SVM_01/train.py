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


y, x = svmutil.svm_read_problem(TRAIN_DATA)
prob = svmutil.svm_problem(y, x)

# rate, param = find_parameters(TRAIN_DATA)

# # C, gamma = utils.get_para(prob)
# C = param['c']
# gamma = param['g']

C = str(2)
gamma = str(2)

param = svmutil.svm_parameter(
    '-s 0 -t 2 -h 0 -c ' + C + ' -g ' + gamma)

m = svmutil.svm_train(prob, param)
svmutil.svm_save_model(MODEL_PATH, m)
