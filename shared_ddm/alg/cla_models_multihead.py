import tensorflow as tf
import numpy as np
from copy import deepcopy
import pdb

np.random.seed(0)
tf.set_random_seed(0)

def _create_weights(size):
    no_layers = len(size) - 1
    no_weights = 0
    for i in range(no_layers):
        no_weights += size[i] * size[i+1] + size[i+1]

    m = tf.Variable(tf.constant(np.zeros([no_weights]), dtype=tf.float32))
    v = tf.Variable(tf.constant(np.zeros([no_weights]), dtype=tf.float32))
    return no_weights, m, v

def _unpack_weights(m, v, size):
        start_ind = 0
        end_ind = 0
        m_weights = []
        m_biases = []
        v_weights = []
        v_biases = []
        no_layers = len(size) - 1
        for i in range(no_layers):
            Din = size[i]
            Dout = size[i+1]
            end_ind += Din * Dout
            m_weights.append(tf.reshape(m[start_ind:end_ind], [Din, Dout]))
            v_weights.append(tf.reshape(v[start_ind:end_ind], [Din, Dout]))
            start_ind = end_ind
            end_ind += Dout
            m_biases.append(m[start_ind:end_ind])
            v_biases.append(v[start_ind:end_ind])
            start_ind = end_ind
        return m_weights, v_weights, m_biases, v_biases


class MFVI_NN(object):
    def __init__(
            self, lower_size, upper_sizes, 
            no_train_samples=10, no_test_samples=100):
        self.lower_size = lower_size
        self.no_tasks = len(upper_sizes)
        self.upper_sizes = upper_sizes
        self.no_train_samples = no_train_samples
        self.no_test_samples = no_test_samples
        # input and output placeholders
        self.x = tf.placeholder(tf.float32, [None, lower_size[0]])
        self.ys = [
            tf.placeholder(tf.float32, [None, upper_size[-1]])
            for upper_size in upper_sizes]
        self.training_size = tf.placeholder(tf.int32)

        self.lower_net = HalfNet(lower_size)
        self.upper_nets = []
        for t, upper_size in enumerate(self.upper_sizes):
            self.upper_nets.append(HalfNet(upper_size))

        self.costs = self._build_costs()
        self.preds = self._build_preds()

    def _build_costs(self):
        kl_lower = self.lower_net.KL_term()
        costs = []
        N = tf.cast(self.training_size, tf.float32)
        for t, upper_net in enumerate(self.upper_nets):
            kl_upper = upper_net.KL_term()
            log_pred = self.log_prediction_fn(
                self.x, self.ys[t], t, self.no_train_samples)
            cost = tf.div(kl_lower + kl_upper, N) - log_pred
            costs.append(cost)
        return costs

    def _build_preds(self):
        preds = []
        for t, upper_net in enumerate(self.upper_nets):
            pred = self.prediction_fn(self.x,  t, self.no_test_samples)
            preds.append(pred)
        return preds

    def prediction_fn(self, inputs, task_idx, no_samples):
        K = no_samples
        inputs_3d = tf.tile(tf.expand_dims(inputs, 0), [K, 1, 1])
        lower_output = self.lower_net.prediction(inputs_3d, K)
        upper_output = self.upper_nets[task_idx].prediction(lower_output, K)
        return upper_output

    def log_prediction_fn(self, inputs, targets, task_idx, no_samples):
        pred = self.prediction_fn(inputs, task_idx, no_samples)
        targets = tf.tile(tf.expand_dims(targets, 0), [self.no_train_samples, 1, 1])
        log_lik = - tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=targets))
        return log_lik

    def init_session(self, task_idx, learning_rate):
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.costs[task_idx])
        # Initializing the variables
        init = tf.global_variables_initializer()
        # launch a session
        self.sess = tf.Session()
        self.sess.run(init)

    def close_session(self):
        self.sess.close()


    def train(self, x_train, y_train, task_idx, prior_lower, prior_upper, 
        no_epochs=1000, batch_size=100, display_epoch=1):
        N = x_train.shape[0]
        if batch_size > N:
            batch_size = N
        sess = self.sess
        costs = []
        feed_dict = {
            self.lower_net.m0: prior_lower[0], 
            self.lower_net.v0: prior_lower[1], 
            self.upper_nets[task_idx].m0: prior_upper[0], 
            self.upper_nets[task_idx].v0: prior_upper[1],
            self.training_size: N}
                
        # Training cycle
        for epoch in range(no_epochs):
            perm_inds = range(x_train.shape[0])
            np.random.shuffle(perm_inds)
            cur_x_train = x_train[perm_inds]
            cur_y_train = y_train[perm_inds]

            avg_cost = 0.
            total_batch = int(np.ceil(N * 1.0 / batch_size))
            # Loop over all batches
            for i in range(total_batch):
                start_ind = i*batch_size
                end_ind = np.min([(i+1)*batch_size, N])
                batch_x = cur_x_train[start_ind:end_ind, :]
                batch_y = cur_y_train[start_ind:end_ind, :]
                feed_dict[self.x] = batch_x
                feed_dict[self.ys[task_idx]] = batch_y
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run(
                    [self.train_step, self.costs[task_idx]], 
                    feed_dict=feed_dict)
                # Compute average loss
                avg_cost += c / total_batch
                # print i, total_batch, c
            # Display logs per epoch step
            if epoch % display_epoch == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
            costs.append(avg_cost)
        print("Optimization Finished!")
        return costs

    def prediction(self, x_test, task_idx, batch_size=1000):
        # Test model
        N = x_test.shape[0]
        batch_size = N if batch_size > N else batch_size
        total_batch = int(np.ceil(N*1.0/batch_size))
        for i in range(total_batch):
            start_ind = i*batch_size
            end_ind = np.min([(i+1)*batch_size, N])
            batch_x = x_test[start_ind:end_ind, :]
            prediction = self.sess.run(
                [self.preds[task_idx]], 
                feed_dict={self.x: batch_x})[0]
            if i == 0:
                predictions = prediction
            else:
                predictions = np.concatenate((predictions, prediction), axis=1)
        return predictions

    def prediction_prob(self, x_test, task_idx, batch_size=1000):
        prob = self.sess.run(
            [tf.nn.softmax(self.prediction(x_test, task_idx, batch_size))], 
            feed_dict={self.x: x_test})[0]
        return prob

    def get_weights(self, task_idx):
        res = self.sess.run(
            [self.lower_net.params, self.upper_nets[task_idx].params])
        return res

    def assign_weights(self, task_idx, lower_weights, upper_weights, 
        lower_transform=np.log, upper_transform=np.log):
        lower_net = self.lower_net
        self.sess.run(
            [lower_net.assign_m_op, lower_net.assign_v_op],
            feed_dict={
                lower_net.new_m: lower_weights[0], 
                lower_net.new_v: lower_transform(lower_weights[1])})

        if not isinstance(task_idx, (list,)):
            task_idx = [task_idx]
            upper_weights = [upper_weights]

        for i, idx in enumerate(task_idx):
            upper_net = self.upper_nets[idx]
            self.sess.run(
                [upper_net.assign_m_op, upper_net.assign_v_op],
                feed_dict={
                    upper_net.new_m: upper_weights[i][0], 
                    upper_net.new_v: upper_transform(upper_weights[i][1])})

    def reset_optimiser(self):
        optimizer_scope = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            "scope/prefix/for/optimizer")
        self.sess.run(tf.initialize_variables(optimizer_scope))


class HalfNet():
    def __init__(self, size, act_func=tf.nn.tanh):
        self.size = size
        self.no_layers = len(size) - 1
        self.act_func = act_func
        
        # creating weights
        self.no_weights, self.m, self.v = _create_weights(self.size)
        self.mw, self.vw, self.mb, self.vb = _unpack_weights(self.m, self.v, self.size)
        self.params = [self.m, self.v]

        self.new_m = tf.placeholder(tf.float32, [self.no_weights])
        self.new_v = tf.placeholder(tf.float32, [self.no_weights])
        self.assign_m_op = tf.assign(self.m, self.new_m)
        self.assign_v_op = tf.assign(self.v, self.new_v)

        # prior as place holder as these can change
        self.m0 = tf.placeholder(tf.float32, [self.no_weights])
        self.v0 = tf.placeholder(tf.float32, [self.no_weights])

    def prediction(self, inputs, no_samples):
        K = no_samples
        N = tf.shape(inputs)[1]
        Din = self.size[0]
        Dout = self.size[-1]
        mw, vw, mb, vb = self.mw, self.vw, self.mb, self.vb
        act = inputs
        for i in range(self.no_layers):
            m_pre = tf.einsum('kni,io->kno', act, mw[i])
            v_pre = tf.einsum('kni,io->kno', act**2.0, tf.exp(vw[i]))
            eps_w = tf.random_normal([K, N, self.size[i+1]], 0.0, 1.0, dtype=tf.float32)
            pre_W = eps_w * tf.sqrt(1e-9 + v_pre) + m_pre
            eps_b = tf.random_normal([K, 1, self.size[i+1]], 0.0, 1.0, dtype=tf.float32)
            pre_b = eps_b * tf.exp(0.5*vb[i]) + mb[i] 
            pre = pre_W + pre_b
            act = self.act_func(pre)
        pre = tf.reshape(pre, [K, N, Dout])
        return pre

    def KL_term(self):
        const_term = -0.5 * self.no_weights
        log_std_diff = 0.5 * tf.reduce_sum(tf.log(self.v0) - self.v)
        mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(self.v) + (self.m0- self.m)**2) / self.v0)
        kl = const_term + log_std_diff + mu_diff_term
        return kl

