import matplotlib
matplotlib.use('Agg') 
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import matplotlib.mlab as mlab
import pdb

def visualise_weights(path=""):
	res_0 = np.load(path + 'weights_0.npz')
	lower_0 = res_0['lower']
	m0 = lower_0[0, :]
	v0 = np.exp(lower_0[1, :])

	res_1 = np.load(path + 'weights_1.npz')
	lower_1 = res_1['lower']
	m1 = lower_1[0, :]
	v1 = np.exp(lower_1[1, :])

	no_hiddens = 256
	in_dim = 784
	in_size = [28, 28]
	no_params = in_dim * no_hiddens
	m0 = m0[:no_params].reshape([in_dim, no_hiddens])
	v0 = v0[:no_params].reshape([in_dim, no_hiddens])
	m1 = m1[:no_params].reshape([in_dim, no_hiddens])
	v1 = v1[:no_params].reshape([in_dim, no_hiddens])
	m0min, m0max = np.min(m0), np.max(m0)
	m1min, m1max = np.min(m1), np.max(m1)
	v0min, v0max = np.min(v0), np.max(v0)
	v1min, v1max = np.min(v1), np.max(v1)

	no_cols = 16
	no_rows = 16
	print "creating figures ..."
	fig0, axs0 = plt.subplots(no_rows, no_cols, figsize=(10, 10))
	fig1, axs1 = plt.subplots(no_rows, no_cols, figsize=(10, 10))
	fig2, axs2 = plt.subplots(no_rows, no_cols, figsize=(10, 10))
	fig3, axs3 = plt.subplots(no_rows, no_cols, figsize=(10, 10))
	# fig4, axs4 = plt.subplots(no_rows, no_cols, figsize=(10, 10))
	# fig5, axs5 = plt.subplots(no_rows, no_cols, figsize=(10, 10))
	# fig6, axs6 = plt.subplots(no_rows, no_cols, figsize=(10, 10))
	# fig7, axs7 = plt.subplots(no_rows, no_cols, figsize=(10, 10))
	fig0.suptitle("mean after task 1")
	fig1.suptitle("variance after task 1")
	fig2.suptitle("mean after task 2")
	fig3.suptitle("variance after task 2")
	# fig4.suptitle("mean diff")
	# fig5.suptitle("variance diff")
	# fig6.suptitle("snr task 1")
	# fig7.suptitle("snr task 2")
	for i in range(no_rows):
		for j in range(no_cols):
			print i, j
			k = i * no_cols + j
			ma = m0[:, k].reshape(in_size)
			va = v0[:, k].reshape(in_size)
			mb = m1[:, k].reshape(in_size)
			vb = v1[:, k].reshape(in_size)

			axs0[i, j].matshow(ma, cmap=matplotlib.cm.binary, vmin=m0min, vmax=m0max)
			axs0[i, j].set_xticks(np.array([]))
			axs0[i, j].set_yticks(np.array([]))
			
			axs1[i, j].matshow(va, cmap=matplotlib.cm.binary, vmin=v0min, vmax=v0max)
			axs1[i, j].set_xticks(np.array([]))
			axs1[i, j].set_yticks(np.array([]))
			
			axs2[i, j].matshow(mb, cmap=matplotlib.cm.binary, vmin=m1min, vmax=m1max)
			axs2[i, j].set_xticks(np.array([]))
			axs2[i, j].set_yticks(np.array([]))
			
			axs3[i, j].matshow(vb, cmap=matplotlib.cm.binary, vmin=v1min, vmax=v1max)
			axs3[i, j].set_xticks(np.array([]))
			axs3[i, j].set_yticks(np.array([]))

			# axs4[i, j].matshow(ma - mb, cmap=matplotlib.cm.binary)
			# axs4[i, j].set_xticks(np.array([]))
			# axs4[i, j].set_yticks(np.array([]))

			# axs5[i, j].matshow(va - vb, cmap=matplotlib.cm.binary)
			# axs5[i, j].set_xticks(np.array([]))
			# axs5[i, j].set_yticks(np.array([]))

			# axs6[i, j].matshow(ma**2 / va, cmap=matplotlib.cm.binary)
			# axs6[i, j].set_xticks(np.array([]))
			# axs6[i, j].set_yticks(np.array([]))

			# axs7[i, j].matshow(mb**2 / vb, cmap=matplotlib.cm.binary)
			# axs7[i, j].set_xticks(np.array([]))
			# axs7[i, j].set_yticks(np.array([]))

	# plt.show()
	fig0.savefig('/tmp/lower_mean_1.pdf')
	fig1.savefig('/tmp/lower_var_1.pdf')
	fig2.savefig('/tmp/lower_mean_2.pdf')
	fig3.savefig('/tmp/lower_var_2.pdf')
	# pdb.set_trace()


def check_weight_pruning(path=""):
	res_0 = np.load(path + 'weights_0.npz')
	lower_0 = res_0['lower']
	m0 = lower_0[0, :]
	v0 = np.exp(lower_0[1, :])

	res_1 = np.load(path + 'weights_1.npz')
	lower_1 = res_1['lower']
	m1 = lower_1[0, :]
	v1 = np.exp(lower_1[1, :])

	res_0 = np.load(path + 'weights_0.npz')
	upper_0 = res_0['upper']
	m2 = upper_0[0, :]
	v2 = np.exp(upper_0[1, :])

	res_1 = np.load(path + 'weights_1.npz')
	upper_1 = res_1['upper']
	m3 = upper_1[0, :]
	v3 = np.exp(upper_1[1, :])

	no_hiddens = 256
	in_dim = 784
	in_size = [28, 28]
	no_params = in_dim * no_hiddens
	m0 = m0[:no_params].reshape([in_dim, no_hiddens])
	v0 = v0[:no_params].reshape([in_dim, no_hiddens])
	m1 = m1[:no_params].reshape([in_dim, no_hiddens])
	v1 = v1[:no_params].reshape([in_dim, no_hiddens])

	no_params = no_hiddens * 2
	m2 = m2[:no_params].reshape([no_hiddens, 2])
	v2 = v2[:no_params].reshape([no_hiddens, 2])
	m3 = m3[:no_params].reshape([no_hiddens, 2])
	v3 = v3[:no_params].reshape([no_hiddens, 2])

	# task 1
	x = np.linspace(-2, 2, 100)
	lims = 10
	for i in range(lims):
		print i
		fig, axs = plt.subplots(2, 1)
		lower_i_mean = m0[:, i]
		lower_i_var = v0[:, i]
		upper_i_mean = m2[i, :]
		upper_i_var = v2[i, :]
		for k in range(in_dim):
			axs[1].plot(x, mlab.normpdf(x, lower_i_mean[k], np.sqrt(lower_i_var[k])))
		for k in range(2):
			axs[0].plot(x, mlab.normpdf(x, upper_i_mean[k], np.sqrt(upper_i_var[k])))

		plt.savefig('/tmp/task_1_unit_%d.pdf'%i)


	x = np.linspace(-2, 2, 100)
	for i in range(lims):
		print i
		fig, axs = plt.subplots(2, 1)
		lower_i_mean = m1[:, i]
		lower_i_var = v1[:, i]
		upper_i_mean = m3[i, :]
		upper_i_var = v3[i, :]
		for k in range(in_dim):
			axs[1].plot(x, mlab.normpdf(x, lower_i_mean[k], np.sqrt(lower_i_var[k])))
		for k in range(2):
			axs[0].plot(x, mlab.normpdf(x, upper_i_mean[k], np.sqrt(upper_i_var[k])))

		plt.savefig('/tmp/task_2_unit_%d.pdf'%i)


if __name__ == "__main__":
	# check_weight_pruning(path='small_init/')
	# visualise_weights(path='small_init/')
	visualise_weights(path='prior_init/')