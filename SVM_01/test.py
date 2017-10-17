'''Testing'''
import svmutil

TEST_DATA = 'data/testing_data_libsvm'
MODEL_PATH = 'model/speech.model'

yt, xt = svmutil.svm_read_problem(TEST_DATA)
m = svmutil.svm_load_model(MODEL_PATH)
p_label, p_acc, p_val = svmutil.svm_predict(yt, xt, m)
