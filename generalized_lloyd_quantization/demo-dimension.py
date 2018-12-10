import time
import numpy as np
from matplotlib import pyplot as plt

from null_uniform import compute_quantization as uni
from generalized_lloyd_LBG import compute_quantization as gl
from optimal_generalized_lloyd_LBG import compute_quantization as opt_gl

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

ndims = 4
random_laplacian_samps = np.random.laplace(scale=10, size=(50000, ndims))
dummy_data = np.copy(random_laplacian_samps)
for i in range(1,ndims):
    dummy_data[:, i] = (np.abs(random_laplacian_samps[:, i-1]) +
                        random_laplacian_samps[:, i])

#######################################################
# We can compare this to the optimal generalized Lloyd
BINWIDTHS = 23 + np.arange(ndims)/ndims
starttime = time.time()
print("Getting initial assignments...")
init_assignments = get_init_assignments_for_lloyd(dummy_data, BINWIDTHS)
init_cword_len = (-1. * np.log2(1. / len(init_assignments)) *
                np.ones((len(init_assignments),)))

print("Time to compute initial assignment points:",
    time.time() - starttime)

opt_gl_nd_apts, opt_gl_nd_assignments, opt_gl_nd_MSE, opt_gl_nd_rate = \
    opt_gl(dummy_data, init_assignments, init_cword_len, lagrange_mult=0.1,
    nn_method='brute_scipy')

print("Time to compute nd (optimal) vector quantization:",
    time.time() - starttime)

print("{}d MSE per dimension".format(ndims),opt_gl_nd_MSE/ndims)
print("{}d rate per dimension".format(ndims),opt_gl_nd_rate/ndims)
