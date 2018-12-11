import time
import pickle
import torch
import numpy as np
from matplotlib import pyplot as plt

from null_uniform import compute_quantization as uni
from generalized_lloyd_LBG import compute_quantization as gl
from optimal_generalized_lloyd_LBG import compute_quantization as opt_gl_numpy
from optimal_generalized_lloyd_LBG_torch import compute_quantization as opt_gl_torch

from utils.clustering import get_cluster_assignments
from analysis_transforms import fista

dict_file = '../../../data/sc_dictionary_8x8_lamda0point1_Field.p'
data_file = '../../../data/two_million_unwhite_centered_patches_8x8.p'
1
patch_dimensions = (8, 8)
sparsity_param = 0.1

patch_dataset = pickle.load(open(data_file, 'rb'))
zero_mean_patches = np.transpose(
    patch_dataset['batched_patches'], (0, 2, 1)).reshape(
        (-1, patch_dimensions[0]*patch_dimensions[1])).T
img_patch_comp_means = patch_dataset['original_patch_means']

device = 'cuda:2'
torch.cuda.set_device(2)

sc_dictionary = pickle.load(open(dict_file, 'rb'))

print('running FISTA')
raw_sc_codes = fista.run(
    torch.from_numpy(zero_mean_patches).to(device),
    torch.from_numpy(sc_dictionary).to(device), sparsity_param, 1000).cpu().numpy()
#^ now samples index first dim, code coefficients in second dim
print('done')

Y = zero_mean_patches.T #(d,n)
A = sc_dictionary       # (n,n)
X = raw_sc_codes.T      # (d,n)
# Note: Y.T = A@X.T

# Use only some X
X = X[:50000]

# Create Clusters
num_clusters = int(A.shape[1]/3)
cluster_assignments = get_cluster_assignments(A, num_clusters)
clusters = [[] for c in range(num_clusters)]
for p in range(A.shape[1]):
    clusters[cluster_assignments[p]].append(p)

def compute_quantization_wrapper(data, quant_method='uni', clusters=None,
                                binwidth=1, placement_scheme='on_mean',
                                lagrange_mult=1., nn_method='brute_break',
                                device='cpu'):
    """
    Parameters
        data: ndarray (d,n)
        quant_method: str {'uni', 'lloyd', 'opt_lloyd'}
        clusters: [cluster1, cluster2, ...] where cluster1 = [idx_1, idx_2, ...]
        binwidth: float
        placement_scheme: str {'on_mode', 'on_median', 'on_mean', 'on_zero'}
        lagrange_mult: float for lloyd
        nn_method: str {'brute_np', 'brute_scipy', 'brute_break', 'kdtree'}
        device: str {'numpy', 'cpu', 'cuda', ...}
    Returns

    """
    init_clusters = clusters
    if clusters is None:
        clusters = [list(range(data.shape[1]))]

    a_pts_all = []
    c_ass_all = []
    MSE_total = 0
    rate_total = 0
    for cluster in clusters:
        cluster_dim = len(cluster)
        print(cluster_dim)
        Xc = data[:,cluster]
        print('Xc.shape',Xc.shape)
        if quant_method=='uni':
            a_pts, c_ass, MSE, rate = uni(Xc, np.array([binwidth]*cluster_dim), placement_scheme=placement_scheme)
        elif quant_method=='lloyd':
            init_apts, _, _, _ = uni(Xc, np.array([binwidth]*cluster_dim), placement_scheme=placement_scheme)
            a_pts, c_ass, MSE, rate = gl(Xc, init_apts, force_const_num_assignment_pts=False)
        elif quant_method=='opt_lloyd':
            init_apts, _, _, _ = uni(Xc, np.array([binwidth]*cluster_dim), placement_scheme=placement_scheme)
            init_cword_len = (-1. * np.log2(1. / len(init_apts)) *np.ones((len(init_apts),)))
            if device=='numpy' or cluster_dim>5:
                a_pts, c_ass, MSE, rate = opt_gl_numpy(Xc, init_apts, init_cword_len, lagrange_mult=lagrange_mult,
                                            nn_method=nn_method)
            else:
                try:
                    a_pts, c_ass, MSE, rate = opt_gl_torch(Xc, init_apts, init_cword_len, lagrange_mult=lagrange_mult,
                                                nn_method=nn_method, device=device)
                except RuntimeError as e:
                    # Cuda mem error; Use numpy
                    print("Runtime error: {}".format(e))
                    print("Switching to numpy")
                    a_pts, c_ass, MSE, rate = opt_gl_numpy(Xc, init_apts, init_cword_len, lagrange_mult=lagrange_mult,
                                                nn_method=nn_method)


        else:
            raise ValueError("Invalid quant_method {}".format(quant_method))
        print(cluster_dim,  MSE, rate)
        a_pts_all.append(a_pts)
        c_ass_all.append(c_ass)
        MSE_total += MSE
        rate_total += rate
    if init_clusters is None:
        a_pts_all = a_pts_all[0]
        c_ass_all = c_ass_all[0]
    return a_pts_all, c_ass_all, MSE_total, rate_total

a_pts, c_ass, MSE, rate = compute_quantization_wrapper(X,clusters=clusters,quant_method='opt_lloyd',
                                                binwidth=8, device=device)

print('MSE',MSE)
print('rate',rate)
