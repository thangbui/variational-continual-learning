import numpy as np
import tensorflow as tf
import utils
from cla_models_multihead import Vanilla_NN, MFVI_NN
from copy import deepcopy
import pdb

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




ide_func = lambda x: np.float32(x)
log_func = lambda x: np.float32(np.log(x))
exp_func = lambda x: np.float32(np.exp(x))

def vectorise(weight_list, no_weights, network_size, transform_func=ide_func):
    weight_vec = np.zeros([no_weights])
    ind = 0
    weight_all = weight_list[0]
    bias_all = weight_list[1]
    weight_last = weight_list[2]
    bias_last = weight_list[3]

    for i in range(len(network_size)-2):
        w = transform_func(np.array(weight_all[i]))
        w_f = w.flatten()
        n_i = w_f.shape[0]
        weight_vec[ind:(ind+n_i)] = w_f
        ind += n_i

        w = transform_func(np.array(bias_all[i]))
        w_f = w.flatten()
        n_i = w_f.shape[0]
        weight_vec[ind:(ind+n_i)] = w_f
        ind += n_i
    
    w = transform_func(np.array(weight_last[0]))
    w_f = w.flatten()
    n_i = w_f.shape[0]
    weight_vec[ind:(ind+n_i)] = w_f
    ind += n_i

    w = transform_func(np.array(bias_last[0]))
    w_f = w.flatten()
    n_i = w_f.shape[0]
    weight_vec[ind:(ind+n_i)] = w_f
    ind += n_i

    return weight_vec

def matricise(weight_vec, no_weights, network_size, transform_func=ide_func):
    no_layers = len(network_size) - 1
    w_mat = []
    b_mat = []
    w_last = []
    b_last = []
    begin_ind = 0
    for i in range(no_layers-1):
        din = network_size[i]
        dout = network_size[i+1]
        end_ind = begin_ind + din * dout
        w = transform_func(weight_vec[begin_ind:end_ind])
        w_mat.append(w.reshape([din, dout]))
        begin_ind = end_ind
        end_ind = begin_ind + dout
        b = transform_func(weight_vec[begin_ind:end_ind])
        b_mat.append(b)
        begin_ind = end_ind
    
    din = network_size[-2]
    dout = network_size[-1]
    end_ind = begin_ind + din * dout
    w = transform_func(weight_vec[begin_ind:end_ind])
    w_last.append(w.reshape([din, dout]))
    begin_ind = end_ind
    end_ind = begin_ind + dout
    b = transform_func(weight_vec[begin_ind:end_ind])
    b_last.append(b)
    begin_ind = end_ind
    
    return w_mat, b_mat, w_last, b_last

def compute_cavity(prior_1, prior_2, data_1, data_2, core_1, core_2, data_idx, core_idx):
    data_n1 = data_1[data_idx, :]
    data_n2 = data_2[data_idx, :]
    core_n1 = core_1[core_idx, :]
    core_n2 = core_2[core_idx, :]
    post_n1 = prior_1 + np.sum(data_n1, axis=0) + np.sum(core_n1, axis=0)
    post_n2 = prior_2 + np.sum(data_n2, axis=0) + np.sum(core_n2, axis=0)
    post_var = 1.0 / post_n2
    post_mean = post_n1 * post_var
    return post_mean, post_var, post_n1, post_n2


def run_vcl_local(hidden_size, no_epochs, data_gen, coreset_method, 
            coreset_size, batch_size=None, single_head=True, no_iters=5):
    in_dim, out_dim = data_gen.get_dims()
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []
    no_weights, network_size = get_no_weights(in_dim, hidden_size, out_dim)
    no_tasks = data_gen.max_iter

    # factors for non-coreset and coreset points
    # mean and variance
    #d_mean = np.zeros([no_tasks, no_weights])
    #d_var = 10**9 * np.ones([no_tasks, no_weights])
    #c_mean = np.zeros([no_tasks, no_weights])
    #c_var = 10**9 * np.ones([no_tasks, no_weights])
    # natural parameters
    d_n1 = np.zeros([no_tasks, no_weights])
    d_n2 = np.zeros([no_tasks, no_weights])
    c_n1 = np.zeros([no_tasks, no_weights])
    c_n2 = np.zeros([no_tasks, no_weights])
    # prior factors
    p_n1 = np.zeros([no_weights])
    p_n2 = np.ones([no_weights])
    
    all_acc = np.array([])

    for task_id in range(no_tasks):
        print 'task %d/%d..' % (task_id, no_tasks)
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)

        # Set the readout head to train
        head = 0
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        # Train network with maximum likelihood to initialize first model
        #if task_id == 0:
        #    print 'training maximum likelihood...'
        #    ml_model = Vanilla_NN(in_dim, hidden_size, out_dim, x_train.shape[0])
        #    ml_model.train(x_train, y_train, task_id, no_epochs, bsize)
        #    # ml_model.train(x_train, y_train, task_id, 1, bsize)
        #    mf_weights = ml_model.get_weights()
        #    mf_variances = None
        #    ml_model.close_session()

        x_coresets, y_coresets, x_train, y_train = coreset_method(
            x_coresets, y_coresets, x_train, y_train, coreset_size)

        for i in range(no_iters):
            print 'task %d/%d, iteration %d/%d..' % (task_id, no_tasks, i, no_iters)
            #pdb.set_trace()
            # finding data factor
            # compute cavity and this is now the prior
            data_idx = range(task_id)
            core_idx = range(task_id+1)
            cav_mean, cav_var, cav_n1, cav_n2 = compute_cavity(
                p_n1, p_n2, d_n1, d_n2, c_n1, c_n2, data_idx, core_idx)
            cav_mean_mat = matricise(cav_mean, no_weights, network_size)
            #with np.errstate(invalid='raise'):
            #    try: 
            cav_var[np.where(cav_var<0)[0]] = 1
            cav_log_var_mat = matricise(cav_var, no_weights, network_size, log_func)
            #    except Exception:
            #        pdb.set_trace()
            # create model and train on non-coreset points
            #if task_id == 0 and i == 0:
            #    mf_model = MFVI_NN(
            #        in_dim, hidden_size, out_dim, x_train.shape[0], 
            #        prev_means=mf_weights, prev_log_variances=mf_variances)
            #else:
            if i == 0:
                mf_weights = cav_mean_mat
                mf_variances = cav_log_var_mat
            mf_model = MFVI_NN(
                in_dim, hidden_size, out_dim, x_train.shape[0], 
                prev_means=cav_mean_mat, prev_log_variances=cav_log_var_mat,
                init_means=mf_weights, init_log_variances=mf_variances)
            mf_model.train(x_train, y_train, head, no_epochs, bsize)
            # mf_model.train(x_train, y_train, head, 1, bsize)
            mf_weights, mf_variances = mf_model.get_weights()

            # divide and get the data factor
            post_m_vec = vectorise(mf_weights, no_weights, network_size)
            post_v_vec = vectorise(mf_variances, no_weights, network_size, exp_func)
            post_n1 = post_m_vec / post_v_vec
            post_n2 = 1.0 / post_v_vec
            d_n1[task_id, :] = post_n1 - cav_n1
            d_n2[task_id, :] = post_n2 - cav_n2

            # loop through the coresets, for each find coreset factor
            if coreset_size > 0:
                no_coresets = len(x_coresets)
                for k in range(no_coresets):
                    # pdb.set_trace()
                    x_coreset_k = x_coresets[k]
                    y_coreset_k = y_coresets[k]
                    # compute cavity and this is now the prior
                    data_idx = range(task_id+1)
                    core_idx = range(task_id+1)
                    core_idx.remove(k)
                    cav_mean, cav_var, cav_n1, cav_n2 = compute_cavity(
                    p_n1, p_n2, d_n1, d_n2, c_n1, c_n2, data_idx, core_idx)
                    cav_mean_mat = matricise(cav_mean, no_weights, network_size)
                    #with np.errstate(invalid='raise'):
                    #    try:
                    cav_var[np.where(cav_var<0)[0]] = 1
                    cav_log_var_mat = matricise(cav_var, no_weights, network_size, log_func)
                    #    except Exception:
                    #        pdb.set_trace()
                    # create model and train on coreset points
                    mf_model = MFVI_NN(
                        in_dim, hidden_size, out_dim, x_coreset_k.shape[0], 
                        prev_means=cav_mean_mat, prev_log_variances=cav_log_var_mat,
                        init_means=mf_weights, init_log_variances=mf_variances)
                    mf_model.train(x_coreset_k, y_coreset_k, head, no_epochs, bsize)
                    # mf_model.train(x_coreset_k, y_coreset_k, head, 1, bsize)
                    mf_weights, mf_variances = mf_model.get_weights()

                    # divide and get the coreset factor
                    post_m_vec = vectorise(mf_weights, no_weights, network_size)
                    post_v_vec = vectorise(mf_variances, no_weights, network_size, exp_func)
                    post_n1 = post_m_vec / post_v_vec
                    post_n2 = 1.0 / post_v_vec
                    c_n1[k, :] = post_n1 - cav_n1
                    c_n2[k, :] = post_n2 - cav_n2

        # test using the final model
        acc = []
        for i in range(len(x_testsets)):
            final_model = mf_model
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
        print acc

    return all_acc

