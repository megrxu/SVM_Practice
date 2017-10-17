'''Testing'''
import svmutil
import numpy as np
import utils

TEST_PRID_DATA = 'data/prid_data_libsvm'
MODEL_PATH = 'model/speech.model'
PREI_DATA = 'data/testing.data'
RESULT = 'output/result.txt'

baredata = np.loadtxt(PRElI_DATA)

all_data = np.zeros([baredata.shape[0], 7])

all_data[:, 0:6] = baredata
all_data[:, 6:7] = -1

mean_list = [1.01461146, 1.02425781, 1.0200618,
             0.29729532, 0.29816894, 0.29703226, -0.0025]
std_list = [1.25313354, 1.29048531, 1.32336985,
            0.12867661, 0.12776173, 0.12808684, 0.99999687]

for i in range(all_data.shape[1] - 1):
    all_data[:, i] -= mean_list[i]
    all_data[:, i] /= std_list[i]

utils.format_data(all_data, TEST_PRID_DATA)

yt, xt = svmutil.svm_read_problem(TEST_PRID_DATA)
m = svmutil.svm_load_model(MODEL_PATH)
p_label, p_acc, p_val = svmutil.svm_predict(yt, xt, m)

np.savetxt(RESULT, np.sign(p_val).astype(int), fmt='%d')
