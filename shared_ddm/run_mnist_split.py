import numpy as np
import tensorflow as tf
import gzip
import cPickle
import sys
sys.path.extend(['alg/'])
import vcl
import coreset
import utils
from copy import deepcopy
import pickle

class SplitMnistGenerator():
    def __init__(self):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.X_test = test_set[0]
        self.train_label = np.hstack((train_set[1], valid_set[1]))
        self.test_label = test_set[1]

        # self.sets_0 = [0, 2, 4, 6, 8]
        # self.sets_1 = [1, 3, 5, 7, 9]
        self.sets_0 = [2, 8]
        self.sets_1 = [3, 9]
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 2

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            # Retrieve train data
            train_0_id = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
            train_1_id = np.where(self.train_label == self.sets_1[self.cur_iter])[0]
            next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))

            next_y_train = np.vstack((np.ones((train_0_id.shape[0], 1)), np.zeros((train_1_id.shape[0], 1))))
            next_y_train = np.hstack((next_y_train, 1-next_y_train))

            # Retrieve test data
            test_0_id = np.where(self.test_label == self.sets_0[self.cur_iter])[0]
            test_1_id = np.where(self.test_label == self.sets_1[self.cur_iter])[0]
            next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))

            next_y_test = np.vstack((np.ones((test_0_id.shape[0], 1)), np.zeros((test_1_id.shape[0], 1))))
            next_y_test = np.hstack((next_y_test, 1-next_y_test))

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

hidden_size = [256]
batch_size = 256
no_epochs = 120
no_iters = 1
coreset_size = 40

# Run vanilla VCL
tf.reset_default_graph()
tf.set_random_seed(12)
np.random.seed(1)


if len(sys.argv) == 2:
    option = int(sys.argv[1])
else:
    option = 4

if option == 1:
    coreset_size = 0
    data_gen = SplitMnistGenerator()
    vcl_result = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen, 
        coreset.rand_from_batch, coreset_size, batch_size, no_iters=no_iters)
    print vcl_result
    pickle.dump(vcl_result, open('results/vcl_split_result_%d.pkl'%no_iters, 'wb'), pickle.HIGHEST_PROTOCOL)

elif option == 2:
    # Run random coreset VCL
    data_gen = SplitMnistGenerator()
    rand_vcl_result = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen, 
        coreset.rand_from_batch, coreset_size, batch_size, no_iters=no_iters)
    print rand_vcl_result
    pickle.dump(rand_vcl_result, open('results/rand_vcl_split_result_%d.pkl'%no_iters, 'wb'), pickle.HIGHEST_PROTOCOL)


elif option == 3:
    # Run k-center coreset VCL
    data_gen = SplitMnistGenerator()
    kcen_vcl_result = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen, 
        coreset.k_center, coreset_size, batch_size, no_iters=no_iters)
    print kcen_vcl_result
    pickle.dump(kcen_vcl_result, open('results/kcen_vcl_split_result_%d.pkl'%no_iters, 'wb'), pickle.HIGHEST_PROTOCOL)


# # Plot average accuracy
# vcl_avg = np.nanmean(vcl_result, 1)
# rand_vcl_avg = np.nanmean(rand_vcl_result, 1)
# kcen_vcl_avg = np.nanmean(kcen_vcl_result, 1)
# utils.plot('results/split.jpg', vcl_avg, rand_vcl_avg, kcen_vcl_avg)
