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
import pdb

class PermutedMnistGenerator():
    def __init__(self, max_iter=10):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.Y_train = np.hstack((train_set[1], valid_set[1]))
        self.X_test = test_set[0]
        self.Y_test = test_set[1]
        self.max_iter = max_iter
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            np.random.seed(self.cur_iter)
            perm_inds = range(self.X_train.shape[1])
            np.random.shuffle(perm_inds)

            # Retrieve train data
            next_x_train = deepcopy(self.X_train)
            next_x_train = next_x_train[:,perm_inds]
            next_y_train = np.eye(10)[self.Y_train]

            # Retrieve test data
            next_x_test = deepcopy(self.X_test)
            next_x_test = next_x_test[:,perm_inds]
            next_y_test = np.eye(10)[self.Y_test]

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

# hidden_size = [100, 100]
batch_size = 500
hidden_size = [100]
no_epochs = 150
single_head = True
num_tasks = 5
coreset_size = 200
num_iters = 1

tf.reset_default_graph()
tf.set_random_seed(12)
np.random.seed(1)

if len(sys.argv) == 2:
    option = int(sys.argv[1])
else:
    option = 4

if option == 1:
    # Run vanilla VCL
    coreset_size = 0
    data_gen = PermutedMnistGenerator(num_tasks)
    vcl_result = vcl.run_vcl_local(hidden_size, no_epochs, data_gen, 
        coreset.rand_from_batch, coreset_size, batch_size, single_head, num_iters)
    print vcl_result
    pickle.dump(vcl_result, open('results/vcl_result_%d.pkl'%num_iters, 'wb'), pickle.HIGHEST_PROTOCO)

elif option == 2:
    # Run random coreset VCL
    data_gen = PermutedMnistGenerator(num_tasks)
    rand_vcl_result = vcl.run_vcl_local(hidden_size, no_epochs, data_gen, 
        coreset.rand_from_batch, coreset_size, batch_size, single_head, num_iters)
    print rand_vcl_result
    pickle.dump(rand_vcl_result, open('results/rand_vcl_result_%d.pkl'%num_iters, 'wb'), pickle.HIGHEST_PROTOCOL)

elif option == 3:
    # Run k-center coreset VCL
    data_gen = PermutedMnistGenerator(num_tasks)
    kcen_vcl_result = vcl.run_vcl_local(hidden_size, no_epochs, data_gen, 
        coreset.k_center, coreset_size, batch_size, single_head, num_iters)
    print kcen_vcl_result
    pickle.dump(kcen_vcl_result, open('results/kcen_vcl_result_%d.pkl'%num_iters, 'wb'), pickle.HIGHEST_PROTOCOL)

if option == 4:
    # load result and plot
    vcl_result = pickle.load(open('results/vcl_result.pkl_%d'%num_iters, 'rb'))
    rand_vcl_result = pickle.load(open('results/rand_vcl_result_%d.pkl'%num_iters, 'rb'))
    kcen_vcl_result = pickle.load(open('results/kcen_vcl_result_%d.pkl'%num_iters, 'rb'))
    # Plot average accuracy
    vcl_avg = np.nanmean(vcl_result, 1)
    rand_vcl_avg = np.nanmean(rand_vcl_result, 1)
    kcen_vcl_avg = np.nanmean(kcen_vcl_result, 1)
    utils.plot('results/permuted_local_%d.pdf'%num_iters, vcl_avg, rand_vcl_avg, kcen_vcl_avg)
