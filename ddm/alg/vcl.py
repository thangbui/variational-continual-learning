import numpy as np
import tensorflow as tf
import utils
from cla_models_multihead import Vanilla_NN, MFVI_NN
from copy import deepcopy

def run_vcl(hidden_size, no_epochs, data_gen, coreset_method, 
            coreset_size=0, batch_size=None, single_head=True):
    in_dim, out_dim = data_gen.get_dims()
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []

    all_acc = np.array([])

    for task_id in range(data_gen.max_iter):
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)

        # Set the readout head to train
        head = 0 if single_head else task_id
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        # Train network with maximum likelihood to initialize first model
        if task_id == 0:
            ml_model = Vanilla_NN(in_dim, hidden_size, out_dim, x_train.shape[0])
            ml_model.train(x_train, y_train, task_id, no_epochs, bsize)
            mf_weights = ml_model.get_weights()
            mf_variances = None
            ml_model.close_session()

        # Select coreset if needed
        if coreset_size > 0:
            x_coresets, y_coresets, x_train, y_train = coreset_method(
                x_coresets, y_coresets, x_train, y_train, coreset_size)

        # Train on non-coreset data
        mf_model = MFVI_NN(
            in_dim, hidden_size, out_dim, x_train.shape[0], 
            prev_means=mf_weights, prev_log_variances=mf_variances)
        mf_model.train(x_train, y_train, head, no_epochs, bsize)
        mf_weights, mf_variances = mf_model.get_weights()

        # Incorporate coreset data and make prediction
        acc = utils.get_scores(
            mf_model, x_testsets, y_testsets, x_coresets, y_coresets, 
            hidden_size, no_epochs, single_head, batch_size)
        all_acc = utils.concatenate_results(acc, all_acc)

        mf_model.close_session()

    return all_acc

def get_no_weights(in_dim, hidden_size, out_dim):
    size = deepcopy(hidden_size)
    size.append(out_dim)
    size.insert(0, in_dim)
    no_weights = 0
    for i in range(len(size) - 1):
        no_weights += (size[i] * size[i+1] + size[i+1])
    return no_weights, size

def vectorise(weight_list, no_weights, network_size):
    weight_vec = np.zeros(weight_vec)
    ind = 0
    weight_all = weight_list[0]
    bias_all = weight_list[1]
    weight_last = weight_list[2]
    bias_last = weight_list[3]

    for i in range(len(network_size)-2):
        w = np.array(weight_all[i])
        w_f = np.flatten(w)
        n_i = w_f.shape[0]
        weight_vec[ind:(ind+n_i)] = w_f
        ind += n_i

        w = np.array(bias_all[i])
        w_f = np.flatten(w)
        n_i = w_f.shape[0]
        weight_vec[ind:(ind+n_i)] = w_f
        ind += n_i
    
    w = np.array(weight_last[0])
    w_f = np.flatten(w)
    n_i = w_f.shape[0]
    weight_vec[ind:(ind+n_i)] = w_f
    ind += n_i

    w = np.array(bias_last[0])
    w_f = np.flatten(w)
    n_i = w_f.shape[0]
    weight_vec[ind:(ind+n_i)] = w_f
    ind += n_i

    return weight_vec

def matricise(weight_vec, no_weights, network_size):
    pass


def run_vcl_local(hidden_size, no_epochs, data_gen, coreset_method, 
            coreset_size, batch_size=None, single_head=True, no_iters=5):
    in_dim, out_dim = data_gen.get_dims()
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []
    no_weights, network_size = get_no_weights(in_dim, hidden_size, out_dim)
    no_tasks = data_gen.max_iter

    # factors for non-coreset and coreset points
    # mean and variance
    d_mean = np.zeros([no_tasks, no_weights])
    d_var = 10**9 * np.ones([no_tasks, no_weights])
    c_mean = np.zeros([no_tasks, no_weights])
    c_var = 10**9 * np.ones([no_tasks, no_weights])
    # natural parameters
    d_n1 = np.zeros([no_tasks, no_weights])
    d_n2 = np.zeros([no_tasks, no_weights])
    c_n1 = np.zeros([no_tasks, no_weights])
    c_n2 = np.zeros([no_tasks, no_weights])
    
    all_acc = np.array([])

    for task_id in range(no_tasks):
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)

        # Set the readout head to train
        head = 0
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        # Train network with maximum likelihood to initialize first model
        if task_id == 0:
            ml_model = Vanilla_NN(in_dim, hidden_size, out_dim, x_train.shape[0])
            # ml_model.train(x_train, y_train, task_id, no_epochs, bsize)
            ml_model.train(x_train, y_train, task_id, no_epochs=10, batch_size=bsize)
            mf_weights = ml_model.get_weights()
            mf_variances = None
            ml_model.close_session()

        x_coresets, y_coresets, x_train, y_train = coreset_method(
            x_coresets, y_coresets, x_train, y_train, coreset_size)

        for i in range(no_iters):
            # finding data factor
            # compute cavity and this is the prior
            # create model and train on non-coreset points
            # divide and get the data factor

            # loop through the coresets, for each find coreset factor
            no_coresets = len(x_coresets)
            for k in range(no_coresets):
                # compute cavity and this is the prior
                # create model and train o coreset points
                # divide and get the data factor

        # test using the final model

        import pdb
        pdb.set_trace()

        # Train on non-coreset data
        mf_model = MFVI_NN(
            in_dim, hidden_size, out_dim, x_train.shape[0], 
            prev_means=mf_weights, prev_log_variances=mf_variances)
        mf_model.train(x_train, y_train, head, no_epochs, bsize)
        mf_weights, mf_variances = mf_model.get_weights()

        # Incorporate coreset data and make prediction
        # acc = utils.get_scores(
        #     mf_model, x_testsets, y_testsets, x_coresets, y_coresets, 
        #     hidden_size, no_epochs, single_head, batch_size)

        mf_weights, mf_variances = model.get_weights()
        acc = []

        if single_head:
            if len(x_coresets) > 0:
                x_train, y_train = merge_coresets(x_coresets, y_coresets)
                bsize = x_train.shape[0] if (batch_size is None) else batch_size
                final_model = MFVI_NN(
                    x_train.shape[1], hidden_size, y_train.shape[1], x_train.shape[0], 
                    prev_means=mf_weights, prev_log_variances=mf_variances)
                final_model.train(x_train, y_train, 0, no_epochs, bsize)
            else:
                final_model = model

        for i in range(len(x_testsets)):
            final_model = model
            head = 0 if single_head else i
            x_test, y_test = x_testsets[i], y_testsets[i]

            pred = final_model.prediction_prob(x_test, head)
            pred_mean = np.mean(pred, axis=0)
            pred_y = np.argmax(pred_mean, axis=1)
            y = np.argmax(y_test, axis=1)
            cur_acc = len(np.where((pred_y - y) == 0)[0]) * 1.0 / y.shape[0]
            acc.append(cur_acc)

        final_model.close_session()

        all_acc = utils.concatenate_results(acc, all_acc)

        mf_model.close_session()

    return all_acc

