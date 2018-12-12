import os
import time
import pickle
import torch
import numpy as np
from matplotlib import pyplot as plt

from null_uniform import compute_quantization as uni
from generalized_lloyd_LBG import compute_quantization as gl
from optimal_generalized_lloyd_LBG import compute_quantization as opt_gl_numpy
from optimal_generalized_lloyd_LBG import calculate_assignment_probabilites
from optimal_generalized_lloyd_LBG_torch import compute_quantization as opt_gl_torch

from utils.clustering import get_clusters
from analysis_transforms import fista

dict_file = '../../../data/sc_dictionary_8x8_lamda0point1_Field.p'
data_file = '../../../data/two_million_unwhite_centered_patches_8x8.p'

# Parameters for FISTA
patch_dimensions = (8, 8)
sparsity_param = 0.1
fista_device = 'cuda:1'
torch.cuda.set_device(1)

patch_dataset = pickle.load(open(data_file, 'rb'))
zero_mean_patches = np.transpose(
    patch_dataset['batched_patches'], (0, 2, 1)).reshape(
        (-1, patch_dimensions[0]*patch_dimensions[1])).T
img_patch_comp_means = patch_dataset['original_patch_means']

sc_dictionary = pickle.load(open(dict_file, 'rb'))

print('running FISTA')
raw_sc_codes = fista.run(
    torch.from_numpy(zero_mean_patches).to(fista_device),
    torch.from_numpy(sc_dictionary).to(fista_device), sparsity_param, 1000).cpu().numpy()
#^ now samples index first dim, code coefficients in second dim
print('done')

torch.cuda.empty_cache()

Y = zero_mean_patches.T #(d,n)
A = sc_dictionary       # (n,n)
X = raw_sc_codes.T      # (d,n)
# Note: Y.T = A@X.T

# Use only some X
X = X[:50000]

# Parameters for quant-computation
quant_method = 'opt_lloyd'
lagrange_mult = 1.
num_bins = 40
num_clusters = 60
clustering_algo = 'stoer_wagner'
quant_device = 'numpy'
# torch.cuda.set_device(1)

def compute_quantization_wrapper(data, quant_method='uni', clusters=None,
                                binwidth=1, placement_scheme='on_mean',
                                lagrange_mult=1., nn_method='brute_break',
                                device='cpu'):
    """
    Parameters
        data: ndarray (d,n)
        quant_method: str {'uni', 'lloyd', 'opt_lloyd'}
        clusters: [cluster1, cluster2, ...] where cluster1 = [idx_1, idx_2, ...]
        binwidth: float or ndarray(n)
        placement_scheme: str {'on_mode', 'on_median', 'on_mean', 'on_zero'}
        lagrange_mult: float
        nn_method: str {'brute_np', 'brute_scipy', 'brute_break', 'kdtree'}
        device: str {'numpy', 'cpu', 'cuda', ...}
    Returns
        a_pts, c_ass, MSE, rate
    """
    data_dim = data.shape[1]
    if clusters is None:
        clusters = [list(range(data_dim))]
    if isinstance(binwidth, np.ndarray):
        assert(binwidth.shape == (data_dim,))
    else:
        binwidth = np.array([float(binwidth)]*data_dim)

    a_pts_all = []
    c_ass_all = []
    MSE_total = 0
    rate_total = 0
    for cluster in clusters:
        cluster_dim = len(cluster)
        Xc = data[:,cluster]
        binwidth_c = binwidth[cluster]
        print('cluster of size {}:'.format(cluster_dim),cluster)
        if quant_method=='uni':
            a_pts, c_ass, MSE, rate = uni(Xc, binwidth_c, placement_scheme=placement_scheme)
        elif quant_method=='lloyd':
            init_apts, _, _, _ = uni(Xc, binwidth_c, placement_scheme=placement_scheme)
            a_pts, c_ass, MSE, rate = gl(Xc, init_apts, force_const_num_assignment_pts=False)
        elif quant_method=='opt_lloyd':
            init_apts, _, _, _ = uni(Xc, binwidth_c, placement_scheme=placement_scheme)
            init_cword_len = (-1. * np.log2(1. / len(init_apts)) *np.ones((len(init_apts),)))
            if device=='numpy':
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
        print('MSE', MSE, 'rate', rate)
        a_pts_all.append(a_pts)
        c_ass_all.append(c_ass)
        MSE_total += MSE
        rate_total += rate
    return a_pts_all, c_ass_all, MSE_total, rate_total

clusters = get_clusters(A, num_clusters, algo=clustering_algo)
binwidths = X.ptp(axis=0)/num_bins
a_pts_all, c_ass_all, MSE, rate = compute_quantization_wrapper(X,clusters=clusters,quant_method=quant_method,
                                                binwidth=binwidths, device=quant_device, lagrange_mult=lagrange_mult)
print('MSE',MSE)
print('rate',rate)

ass_probs_all = []
codeword_lengths = []
# Compute codeword lengths and assignment_probablities
for i in range(len(clusters)):
    a_pts = a_pts_all[i]
    c_ass = c_ass_all[i]
    probs = calculate_assignment_probabilites(c_ass, len(a_pts))
    ass_probs_all.append(probs)
    codeword_lengths.append(-1 * np.log2(probs))

# Save to disk
quantization_data = {
    'quant_method':quant_method,
    'dimension':X.shape[1],
    'clusters':clusters,
    'assignment_points':a_pts_all,
    'codeword_lengths':codeword_lengths,
    'trained_on': {
        'dict_file':os.path.basename(dict_file),
        'data_file':os.path.basename(data_file),
        'num_bins':num_bins,
        'binwidths':binwidths,
        'codes_assigned':c_ass_all,
        'clustering_algo':clustering_algo,
        'lagrange_mult':lagrange_mult,
        'MSE':MSE,
        'rate':rate
    }
}
if quant_method != 'uni':
    quantization_code_file = 'quant_codes/quantization_code__{}D__{}__{}_{}_clusters__{}bins__{}MSE__{}RATE.p'.format(X.shape[1], quant_method, num_clusters, clustering_algo, num_bins, MSE, rate)
else:
    quantization_code_file = 'quant_codes/quantization_code__{}D__{}__{}_{}_clusters__{}bins__{}lambda__{}MSE__{}RATE.p'.format(X.shape[1], quant_method, num_clusters, clustering_algo, num_bins, lagrange_mult, MSE, rate)
pickle.dump(quantization_data, open(quantization_code_file, 'wb'))

# Plotting...
num_clusters = len(clusters)
code_sizes = [a_pts.shape[0] for a_pts in a_pts_all]
max_codesize = max(code_sizes)

ass_probs_heatmap = np.zeros((max_codesize,num_clusters))
ass_probs_heatmap.fill(np.nan)
for i in range(num_clusters):
    ass_probs_heatmap[:code_sizes[i],i] = ass_probs_all[i]

c_ass_ndarray = np.array(c_ass_all).T

for _ in range(2):
    # Overlay probablity heatmap with code
    random_data_pt = np.random.randint(0,X.shape[0])
    data_img = (Y[random_data_pt] + img_patch_comp_means).reshape(patch_dimensions)
    data_sc = X[random_data_pt]
    data_sc_clustered = [np.linalg.norm(data_sc[cluster]) for cluster in clusters]
    data_code = c_ass_ndarray[random_data_pt]

    fig = plt.figure(figsize=(15,8))
    plt.imshow(ass_probs_heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Visualizing random data-point #{}".format(random_data_pt))
    plt.plot(c_ass_ndarray[random_data_pt],'ro', color = 'b')
    plt.plot(data_sc_clustered,color = 'r', linestyle='-',linewidth = 1)
    fig.savefig('spare_quant_viz/{}/{}.png'.format(quant_method,random_data_pt))
    plt.close(fig)
