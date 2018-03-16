import numpy as np
import tensorflow as tf
import utils
from cla_models_multihead import MFVI_NN
from copy import deepcopy
import pdb

ide_func = lambda x: np.float32(x)
log_func = lambda x: np.float32(np.log(x))
exp_func = lambda x: np.float32(np.exp(x))

class FactorManager():
    def __init__(self, no_tasks, no_lower_weights, no_upper_weights):
        # natural parameters
        # non-coreset factors
        self.dl_n1 = np.zeros([no_tasks, no_lower_weights])
        self.dl_n2 = np.zeros([no_tasks, no_lower_weights])
        self.du_n1 = [np.zeros([no_weights]) for no_weights in no_upper_weights]
        self.du_n2 = [np.zeros([no_weights]) for no_weights in no_upper_weights]
        # coreset factors
        self.cl_n1 = np.zeros([no_tasks, no_lower_weights])
        self.cl_n2 = np.zeros([no_tasks, no_lower_weights])
        self.cu_n1 = [np.zeros([no_weights]) for no_weights in no_upper_weights]
        self.cu_n2 = [np.zeros([no_weights]) for no_weights in no_upper_weights]
        # prior factors
        self.pl_n1 = np.zeros([no_lower_weights])
        self.pl_n2 = np.ones([no_lower_weights])
        self.pu_n1 = [np.zeros([no_weights]) for no_weights in no_upper_weights]
        self.pu_n2 = [np.ones([no_weights]) for no_weights in no_upper_weights]

    def compute_dist(self, dl_idx, cl_idx, task_idx, remove_data, remove_core):
        dl_n1 = np.sum(self.dl_n1[dl_idx, :], axis=0)
        dl_n2 = np.sum(self.dl_n2[dl_idx, :], axis=0)
        cl_n1 = np.sum(self.cl_n1[cl_idx, :], axis=0)
        cl_n2 = np.sum(self.cl_n2[cl_idx, :], axis=0)
        l_n1 = self.pl_n1 + dl_n1 + cl_n1
        l_n2 = self.pl_n2 + dl_n2 + cl_n2
        l_v = 1.0 / l_n2
        l_m = l_v * l_n1

        u_n1, u_n2, u_m, u_v = [], [], [], []
        for i in task_idx:
            du_n1 = self.du_n1[i]
            du_n2 = self.du_n2[i]
            cu_n1 = self.cu_n1[i]
            cu_n2 = self.cu_n2[i]
            pu_n1 = self.pu_n1[i]
            pu_n2 = self.pu_n2[i]
            u_n1_i = pu_n1
            u_n2_i = pu_n2 
            if not remove_core:
                u_n1_i += cu_n1
                u_n2_i += cu_n2
            if not remove_data:
                u_n1_i += du_n1
                u_n2_i += du_n2
            u_v_i = 1.0 / u_n2_i
            u_m_i = u_v_i * u_n1_i
            
            u_n1.append(u_n1_i)
            u_n2.append(u_n2_i)
            u_m.append(u_m_i)
            u_v.append(u_v_i)
        return (l_m, l_v, l_n1, l_n2), (u_m, u_v, u_n1, u_n2)

    def update_factor(self, post_l_mv, post_u_mv, cav_l_n, cav_u_n, 
        task_idx, data_factor, core_factor, transform_func=np.exp):
        post_l_m, post_l_v = post_l_mv[0], transform_func(post_l_mv[1])
        post_u_m, post_u_v = post_u_mv[0], transform_func(post_u_mv[1])
        post_l_n1, post_l_n2 = post_l_m / post_l_v, 1.0 / post_l_v
        post_u_n1, post_u_n2 = post_u_m / post_u_v, 1.0 / post_u_v
        f_l_n1 = post_l_n1 - cav_l_n[0]
        f_l_n2 = post_l_n2 - cav_l_n[1]
        f_u_n1 = post_u_n1 - cav_u_n[0]
        f_u_n2 = post_u_n2 - cav_u_n[1]       
        if data_factor:
            self.dl_n1[task_idx, :] = f_l_n1
            self.dl_n2[task_idx, :] = f_l_n2
            self.du_n1[task_idx] = f_u_n1
            self.du_n2[task_idx] = f_u_n2
        else:
            self.cl_n1[task_idx, :] = f_l_n1
            self.cl_n2[task_idx, :] = f_l_n2
            self.cu_n1[task_idx] = f_u_n1
            self.cu_n2[task_idx] = f_u_n2


def run_vcl_shared(hidden_size, no_epochs, data_gen, coreset_method, 
            coreset_size=0, batch_size=None, no_iters=1, learning_rate=0.005):
    in_dim, out_dim = data_gen.get_dims()
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []
    x_trainsets, y_trainsets = [], []

    all_acc = np.array([])
    no_tasks = data_gen.max_iter
    for i in range(no_tasks):
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_trainsets.append(x_train)
        y_trainsets.append(y_train)
        x_testsets.append(x_test)
        y_testsets.append(y_test)

    # creating model
    lower_size = [in_dim] + deepcopy(hidden_size)
    upper_sizes = [[hidden_size[-1], out_dim] for i in range(no_tasks)]
    model = MFVI_NN(lower_size, upper_sizes)
    no_lower_weights = model.lower_net.no_weights
    no_upper_weights = [net.no_weights for net in model.upper_nets]
    factory = FactorManager(no_tasks, no_lower_weights, no_upper_weights)

    for task_id in range(no_tasks):
        # init model
        model.init_session(task_id, learning_rate)
        # get data
        x_train, y_train = x_trainsets[task_id], y_trainsets[task_id]
        x_test, y_test = x_testsets[task_id], y_testsets[task_id]

        # Set the readout head to train
        head = task_id
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        # Select coreset if needed
        if coreset_size > 0:
            x_coresets, y_coresets, x_train, y_train = coreset_method(
                x_coresets, y_coresets, x_train, y_train, coreset_size)

        for i in range(no_iters):
            # finding data factor
            # compute cavity
            lower_data_idx = range(task_id)
            lower_core_idx = range(task_id+1)
            lower_cav, upper_cav = factory.compute_dist(
                lower_data_idx, lower_core_idx,
                [task_id], remove_core=False, remove_data=True)
            lower_mv = [lower_cav[0], lower_cav[1]]
            upper_mv = [upper_cav[0][0], upper_cav[1][0]]
            lower_n = [lower_cav[2], lower_cav[3]]
            upper_n = [upper_cav[2][0], upper_cav[3][0]]
            if task_id == 0 and i == 0:
                lower_post, upper_post = lower_mv, upper_mv
                transform_func = log_func
            else:
                transform_func = ide_func
            model.assign_weights(task_id, lower_post, upper_post, transform_func)
            # train on non-coreset data
            model.reset_optimiser()
            model.train(x_train, y_train, task_id, lower_mv, upper_mv, no_epochs, bsize)
            # get params and update factor
            lower_post, upper_post = model.get_weights(task_id)
            factory.update_factor(lower_post, upper_post, lower_n, upper_n, task_id, 
                data_factor=True, core_factor=False)

            # loop through the coresets, for each find coreset factor
            if coreset_size > 0:
                no_coresets = len(x_coresets)
                for k in range(no_coresets):
                    # pdb.set_trace()
                    x_coreset_k = x_coresets[k]
                    y_coreset_k = y_coresets[k]
                    # compute cavity and this is now the prior
                    lower_data_idx = range(task_id+1)
                    lower_core_idx = range(task_id+1)
                    lower_core_idx.remove(k)
                    lower_cav, upper_cav = factory.compute_dist(
                        lower_data_idx, lower_core_idx,
                        [task_id], remove_core=True, remove_data=False)
                    lower_mv = [lower_cav[0], lower_cav[1]]
                    upper_mv = [upper_cav[0][0], upper_cav[1][0]]
                    lower_n = [lower_cav[2], lower_cav[3]]
                    upper_n = [upper_cav[2][0], upper_cav[3][0]]
                    model.assign_weights(task_id, lower_post, upper_post, ide_func)
                    # train on coreset data
                    model.reset_optimiser()
                    # model.train(x_coreset_k, y_coreset_k, task_id, lower_mv, upper_mv, no_epochs, bsize)
                    model.train(x_coreset_k, y_coreset_k, task_id, lower_mv, upper_mv, 100, bsize)
                    # get params and update factor
                    lower_post, upper_post = model.get_weights(task_id)
                    factory.update_factor(lower_post, upper_post, lower_n, upper_n, task_id, 
                        data_factor=False, core_factor=True)

            
        # Make prediction
        lower_post, upper_post = factory.compute_dist(
            range(no_tasks), range(no_tasks), range(no_tasks), False, False)
        lower_mv = [lower_post[0], lower_post[1]]
        upper_mv = [[upper_post[0][i], upper_post[1][i]] for i in range(no_tasks)]
        model.assign_weights(range(no_tasks), lower_mv, upper_mv, log_func)
        # pdb.set_trace()
        acc = utils.get_scores(model, x_testsets, y_testsets)
        all_acc = utils.concatenate_results(acc, all_acc)
        # print acc
        # pdb.set_trace()
        model.close_session()

    return all_acc

def get_no_weights(in_dim, hidden_size, out_dim):
    size = deepcopy(hidden_size)
    size.append(out_dim)
    size.insert(0, in_dim)
    no_weights = 0
    for i in range(len(size) - 1):
        no_weights += (size[i] * size[i+1] + size[i+1])
    return no_weights, size


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

