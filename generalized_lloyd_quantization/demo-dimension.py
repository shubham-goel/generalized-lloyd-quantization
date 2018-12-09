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
        assgnmnts = np.delete(assgnmnts, np.where(assgnmnts < 0.9*min_data))
        assgnmnts = np.delete(assgnmnts, np.where(assgnmnts > 0.9*max_data))
        return assgnmnts
    else:
        mask_min = np.any(assgnmnts[:,:] < 0.9*min_data[None, :], axis=1)
        mask_max = np.any(assgnmnts[:,:] > 0.9*max_data[None, :], axis=1)
        assgnmnts = np.delete(assgnmnts, np.where(mask_min + mask_max), axis=0)
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
print("getting initial assignments...")
init_assignments = get_init_assignments_for_lloyd(dummy_data, BINWIDTHS)
init_cword_len = (-1. * np.log2(1. / len(init_assignments)) *
                np.ones((len(init_assignments),)))

print("Time to compute initial assignment points:",
    time.time() - starttime)

opt_gl_nd_apts, opt_gl_nd_assignments, opt_gl_nd_MSE, opt_gl_nd_rate = \
    opt_gl(dummy_data, init_assignments, init_cword_len, lagrange_mult=0.1)

print("Time to compute nd (optimal) vector quantization:",
    time.time() - starttime)

print("opt_gl_{}d_MSE per dimension".format(ndims),opt_gl_nd_MSE/ndims)
print("opt_gl_{}d_rate per dimension".format(ndims), opt_gl_nd_rate/ndims)
