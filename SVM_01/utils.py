'''Utilities'''
import numpy as np


def load_data(filename, training_size):
    '''Format the training data to a libsvm data structure'''

    # Load data and randomization
    all_data = np.loadtxt(filename)
    all_data = np.random.permutation(all_data)

    # Top rows as training data
    training_data = all_data[:training_size][:]

    # Use training data to normalize the data
    mean_list = np.mean(training_data, 0)
    std_list = np.std(training_data, 0)
    
    for i in range(all_data.shape[1] - 1):
        all_data[:, i] -= mean_list[i]
        all_data[:, i] /= std_list[i]

    # Other data as testing data
    testing_data = all_data[training_size:][:]

    return training_data, testing_data


def format_data(data, filename):
    '''Format the training data to a libsvm data structure'''
    string_data = np.array(data)
    i = data.shape[1]
    string_data[:,1:i] = data[:,0:i-1]
    string_data[:,0] = data[:,i-1]
    string_data = string_data.astype(str)

    for j in range(1, i):
        string_data[:,j] = np.char.add(str(j) + ':', string_data[:,j])
    np.savetxt(filename, string_data, delimiter=' ', fmt='%s')
    return 0
