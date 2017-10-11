'''Traing'''
import utils
import svmutil

TRAIN_DATA = 'data/training_data_libsvm'
TEST_DATA = 'data/testing_data_libsvm'
BARE_DATA = 'data/training.data'
MODEL_PATH = 'model/speech.model'

training_data, testing_data = utils.load_data(BARE_DATA, 5000)
utils.format_data(training_data, TRAIN_DATA)
utils.format_data(testing_data, TEST_DATA)

y, x = svmutil.svm_read_problem(TRAIN_DATA)
prob  = svmutil.svm_problem(y, x)
# param = svmutil.svm_parameter('-s 0 -t 2 -c 8 -h 0 -g 1')
param = svmutil.svm_parameter('-s 0 -t 2 -c 8 -h 0 -g 1 -v 5')
m = svmutil.svm_train(prob, param)
# svmutil.svm_save_model(MODEL_PATH, m)