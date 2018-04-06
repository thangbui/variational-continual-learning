import numpy as np
import tensorflow as tf
import utils
from cla_models_multihead import MFVI_NN, ML_NN
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

        # l_v[np.where(l_v<0)[0]] = 1.0
        # l_n2[np.where(l_n2<0)[0]] = 1.0

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

            u_v_i[np.where(u_v_i<0)[0]] = 1.0
            u_n2_i[np.where(u_n2_i<0)[0]] = 1.0
            
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

def init_post(cav_info, init_using_cav, ml_weights=None):
    if init_using_cav or ml_weights is None:
        return cav_info
    else:
        cav_mean = cav_info[0]
        cav_var = cav_info[1]
        if ml_weights:
            post_mean = ml_weights[0]
        else:  
            post_mean = np.random.normal(size=cav_mean.shape, scale=0.1)
        post_var = np.ones_like(cav_var)*np.exp(-6.0)
        return [post_mean, post_var]

def run_vcl_shared(hidden_size, no_epochs, data_gen, coreset_method, 
            coreset_size=0, batch_size=None, no_iters=1, learning_rate=0.005, 
            init_using_cav=False):
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
    # we also create a model trained using maximum likelihood
    ml_model = ML_NN(lower_size, upper_sizes)
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
                # train a network using maximum likelihood
                ml_model.init_session(task_id, learning_rate=0.002)
                ml_model.train(x_train, y_train, task_id, 
                    no_epochs=50, batch_size=bsize)
                ml_lower, ml_upper = ml_model.get_weights(task_id)

                lower_post = init_post(lower_mv, init_using_cav, ml_lower)
                upper_post = init_post(upper_mv, init_using_cav, ml_upper)
                # lower_post = init_post(lower_mv, init_using_cav)
                # upper_post = init_post(upper_mv, init_using_cav)
                # lower_post, upper_post = lower_mv, upper_mv
                upper_transform = log_func
                lower_transform = log_func
            elif i == 0:
                # upper_post = upper_mv
                upper_post = init_post(upper_mv, init_using_cav)
                upper_transform = log_func
                lower_transform = ide_func
            else:
                upper_transform = ide_func
                lower_transform = ide_func
            model.assign_weights(
                task_id, lower_post, upper_post, lower_transform, upper_transform)
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
                    model.assign_weights(task_id, lower_post, upper_post, ide_func, ide_func)
                    # train on coreset data
                    model.reset_optimiser()
                    model.train(x_coreset_k, y_coreset_k, task_id, lower_mv, upper_mv, no_epochs, bsize)
                    # model.train(x_coreset_k, y_coreset_k, task_id, lower_mv, upper_mv, 100, bsize)
                    # get params and update factor
                    lower_post, upper_post = model.get_weights(task_id)
                    factory.update_factor(lower_post, upper_post, lower_n, upper_n, task_id, 
                        data_factor=False, core_factor=True)

        np.savez('results/weights_%d.npz' % task_id, lower=lower_post, upper=upper_post)


        # Make prediction
        lower_post, upper_post = factory.compute_dist(
            range(no_tasks), range(no_tasks), range(no_tasks), False, False)
        lower_mv = [lower_post[0], lower_post[1]]
        upper_mv = [[upper_post[0][i], upper_post[1][i]] for i in range(no_tasks)]
        model.assign_weights(range(no_tasks), lower_mv, upper_mv, log_func, log_func)
        # pdb.set_trace()
        acc = utils.get_scores(model, x_testsets, y_testsets)
        if task_id == 0:
            all_acc = np.array(acc)
        else:
            all_acc = np.vstack([all_acc, acc])
        print acc
        print all_acc
        # all_acc = utils.concatenate_results(acc, all_acc)
        # pdb.set_trace()
        model.close_session()

    return all_acc