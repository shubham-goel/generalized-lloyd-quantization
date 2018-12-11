
import time
import pickle
import torch
import numpy as np
from matplotlib import pyplot as plt

from null_uniform import compute_quantization as uni
from generalized_lloyd_LBG import compute_quantization as gl
from optimal_generalized_lloyd_LBG import compute_quantization as opt_gl_numpy
from optimal_generalized_lloyd_LBG_torch import compute_quantization as opt_gl_torch

def get_init_assignments_for_lloyd(data, the_binwidths):
    # Lloyd can run into trouble if the most extreme assignment points are
    # larger in magnitude than the most extreme datapoints, which can happen
    # with the uniform quantization, so we just get rid of those initial points.
    assgnmnts, _, _, _ = uni(data, the_binwidths, placement_scheme='on_mean')
    min_data = np.min(data, axis=0)
    max_data = np.max(data, axis=0)
    if data.ndim == 1:
        min_coeff = np.where(min_data<0, 0.9, 1.1)
        max_coeff = np.where(max_data<0, 1.1, 0.9)
        min_mask  = (assgnmnts <= min_coeff*min_data)
        max_mask  = (assgnmnts >= max_coeff*max_data)
        zero_mask = min_mask + max_mask
    else:
        min_coeff = np.where(min_data<0, 0.9, 1.1)
        max_coeff = np.where(max_data<0, 1.1, 0.9)
        min_mask = np.any(assgnmnts[:,:] <= (min_coeff*min_data)[None, :], axis=1)
        max_mask = np.any(assgnmnts[:,:] >= (max_coeff*max_data)[None, :], axis=1)
        zero_mask = min_mask + max_mask
    num_apts_orig = assgnmnts.shape[0]
    num_apts_new = assgnmnts.shape[0] - len(zero_mask.nonzero()[0])
    assgnmnts = np.delete(assgnmnts, np.where(zero_mask), axis=0)
    print("Trimmed extreme assignment points: {} -> {}".format(num_apts_orig, num_apts_new))
    return assgnmnts

DATA_DIM = 64
QUANT_DIM = 64
for QUANT_DIM in [1,2,4,8,16,32,64]:
	print("***************************")
	print("********    QUANT_DIM = {}".format(QUANT_DIM))
	print("***************************")
	assert(DATA_DIM % QUANT_DIM == 0)
	samples_fname = 'dummy_{}D_data.bin'.format(DATA_DIM)
	device = 'cuda:1'
	torch.cuda.set_device(1)

	try:
		dummy_data = pickle.load(open(samples_fname, 'rb'))
	except (OSError, IOError) as e:
		random_laplacian_samps = np.random.laplace(scale=10, size=(50000, DATA_DIM))
		dummy_data = np.copy(random_laplacian_samps)
		for i in range(1,DATA_DIM):
			dummy_data[:, i] = (np.abs(random_laplacian_samps[:, i-1]) +
								random_laplacian_samps[:, i])
		pickle.dump(dummy_data, open(samples_fname, 'wb'))

	# # next uniform vector (Nd)
	uni_2d_rates = []
	uni_2d_MSEs = []
	uni_binwidth = list(np.linspace(8, 32, 50))
	for binwidth in uni_binwidth:
		print('RD curve, vector uniform, binwidth=', binwidth)
		uni_2d_MSE = []
		uni_2d_rate = []
		for cluster in range(int(DATA_DIM/QUANT_DIM)):
			cluster_data = dummy_data[:,cluster*QUANT_DIM:(cluster+1)*QUANT_DIM]
			_, _, uni_2d_MSEc, uni_2d_ratec = uni(
						cluster_data, np.array([binwidth]*QUANT_DIM), placement_scheme='on_mean')
			uni_2d_MSE.append(uni_2d_MSEc)
			uni_2d_rate.append(uni_2d_ratec)
		uni_2d_MSE = np.sum(np.array(uni_2d_MSE))
		uni_2d_rate = np.sum(np.array(uni_2d_rate))
		uni_2d_MSEs.append(uni_2d_MSE / DATA_DIM)
		uni_2d_rates.append(uni_2d_rate / DATA_DIM)

	# # next suboptimal generalized Lloyd (2d)
	# # Inefficient
	# gl_2d_rates = []
	# gl_2d_MSEs = []
	# for binwidth in np.linspace(8, 60, 50):
	# 	print('RD curve, suboptimal vector Lloyd, binwidth=', binwidth)
	# 	init_assignments, _, _, _ = uni(dummy_data, np.array([binwidth]*DATA_DIM),
	# 								placement_scheme='on_mean')
	# 	_, _, gl_2d_MSE, gl_2d_rate = gl(dummy_data, init_assignments,
	# 									force_const_num_assignment_pts=False)
	# 	#^ make this correspond to optimal lloyd with lambda=0.0.
	# 	gl_2d_rates.append(gl_2d_rate / DATA_DIM)
	# 	gl_2d_MSEs.append(gl_2d_MSE / DATA_DIM)

	# finally, the optimal generalized Lloyd
	opt_gl_2d_rates = []
	opt_gl_2d_MSEs = []
	opt_gl_2d_binwidth = 8
	opt_gl_2d_lagrange_w = list(np.linspace(0.0, 4.0, 50))
	for lagrange_w in opt_gl_2d_lagrange_w:
		print('RD curve, optimal vector Lloyd, lagrange mult=', lagrange_w)
		opt_gl_2d_MSE = []
		opt_gl_2d_rate = []
		for cluster in range(int(DATA_DIM/QUANT_DIM)):
			cluster_data = dummy_data[:,cluster*QUANT_DIM:(cluster+1)*QUANT_DIM]

			# init_assignments, _, _, _ = uni(dummy_data, np.array([opt_gl_2d_binwidth]*QUANT_DIM),
			# 									placement_scheme='on_mean')
			init_assignments = get_init_assignments_for_lloyd(cluster_data, np.array([opt_gl_2d_binwidth]*QUANT_DIM))
			init_cword_len = (-1. * np.log2(1. / len(init_assignments)) *
												np.ones((len(init_assignments),)))

			# Try to run on GPU; Run on CPU if fails (out of Memory)
			try:
				_, _, opt_gl_2d_MSEc, opt_gl_2d_ratec = opt_gl_torch(
						cluster_data, init_assignments, init_cword_len, lagrange_mult=lagrange_w,
						device=device, nn_method='brute_break')
			except RuntimeError as e:
				print("CUDA RuntimeError, Switching to Numpy")
				_, _, opt_gl_2d_MSEc, opt_gl_2d_ratec = opt_gl_numpy(
						cluster_data, init_assignments, init_cword_len, lagrange_mult=lagrange_w,
						nn_method='brute_break')

			opt_gl_2d_MSE.append(opt_gl_2d_MSEc)
			opt_gl_2d_rate.append(opt_gl_2d_ratec)
		opt_gl_2d_MSE = np.sum(np.array(opt_gl_2d_MSE))
		opt_gl_2d_rate = np.sum(np.array(opt_gl_2d_rate))
		opt_gl_2d_rates.append(opt_gl_2d_rate / DATA_DIM)
		opt_gl_2d_MSEs.append(opt_gl_2d_MSE / DATA_DIM)

	rd_data = {
		'DATA_DIM': DATA_DIM,
		'QUANT_DIM': QUANT_DIM,
		'uni_vec_mse': uni_2d_MSEs,
		'uni_vec_rate': uni_2d_rates,
		'uni_vec_binwidth': uni_binwidth,
		'gl_vec_mse': None, # gl_2d_MSEs,
		'gl_vec_rate': None, # gl_2d_rates,
		'optgl_vec_mse': opt_gl_2d_MSEs,
		'optgl_vec_rate': opt_gl_2d_rates,
		'optgl_vec_lagrange_w': opt_gl_2d_lagrange_w,
		'optgl_vec_binwidth': opt_gl_2d_binwidth,
	}
	rd_data_fname = 'rd_{}D_cluster_{}D_data.bin'.format(QUANT_DIM, DATA_DIM)
	pickle.dump(rd_data, open(rd_data_fname, 'wb'))

	# plot the three ND variants
	plt.figure(figsize=(20, 20))
	plt.plot(uni_2d_MSEs, uni_2d_rates, label='Uniform {}D'.format(QUANT_DIM), linewidth=4)
	# plt.plot(gl_2d_MSEs, gl_2d_rates, label='Suboptimal {}D Lloyd'.format(QUANT_DIM), linewidth=4)
	plt.plot(opt_gl_2d_MSEs, opt_gl_2d_rates, label='Optimal {}D Lloyd'.format(QUANT_DIM),
						linewidth=4)
	plt.legend(fontsize=15)
	plt.title('Rate-distortion performance of different {}D vector quantization '.format(QUANT_DIM) +
						'schemes', fontsize=20)
	plt.xlabel('Distortion (Mean squared error)', fontsize=15)
	plt.ylabel('Rate (bits per component)', fontsize=15)
plt.show()
