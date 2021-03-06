"""
Most basic null-model alternative to Lloyd quantization, uniformly spaced bins.

We can place assignment points uniformly in an n-dimensional space and then
apply the quantizations based on nearest neighbor assignments. The one choice
that remains to be specified is the precise offset or 'phase' of the assignment
points. One thing that makes some sense is to place an assignment point directly
on the mode, median, or mean of the distribution, which we give the option of
in the function below
"""
import numpy as np
# import hdmedians

def compute_quantization(samples, binwidth, placement_scheme='on_mode'):
  """
  Calculates the assignment points for uniformly-spaced quantization bins

  The problem we need to solve is: given that we have bins with uniform spacing
  (an therefore fixed width), how should they be aligned? Should we place an
  assignment point directly on the mean of the distribution? On the mode of
  the distribution? On the median? On the value zero? This function calculates
  the assignment points based on one of these choices.

  Parameters
  ----------
  samples : ndarray (d, n) or (d,)
      This is an array of d samples of an n-dimensional random variable
      that we wish to find the uniform quantizer for. If these are scalar random
      variables, we will accept a 1d array as input.
  binwidth : ndarray (n, ) or float
      The width of the quantization bins in each dimension. If the input is
      multivariate (samples.ndim = 2), then we must specify a binwidth for each
      of the dimensions.
  placement_scheme : str, optional
      Determines where we place one of the assignment points. It can be one of
      {'on_mode', 'on_median', 'on_mean', 'on_zero'}.
      'on_mode': estimating the distribution from a histogram, take the mode
        of this estimate and place a point directly on this value.
      'on_median': place a point directly on the median of these values.
      'on_mean': place a point directly on the mean of these values.
      'on_zero': place a point directly on the value 0.0.
      Default 'on_mode'.

  Returns
  -------
  assignment_pts : ndarray (m, n) or (m,)
      The converged assignment points
  cluster_assignments : ndarray (d, )
      For each sample, the index of the codeword to which uniform quantization
      assigns this datapoint. We can compute the actual quantized values outside
      this function by invoking `assignment_pts[cluster_assignments]`
  MSE : float
      The mean squared error (the mean l2-normed-squared to be precise) for the
      returned quantization.
  shannon_entropy : float
      The (empirical) Shannon entropy for this code. We can say that assuming
      we use a lossless binary source code, that our expected codeword length
      is precisely this value.
  """
  if samples.ndim == 2:
    assert type(binwidth) == np.ndarray
    assert len(binwidth) == samples.shape[1]
  if placement_scheme == 'on_mode':
    assert samples.shape[0] > 1000, (
        'Cannot accurately estimate the mode of the ' +
        'distribution with so few samples. Try another placement scheme')

  if placement_scheme == 'on_mode':
    if samples.ndim == 1:
      # numpy's histogramdd() is slow on 1d samples for some reason so we
      # use good old-fashioned histogram()
      counts, hist_bin_edges = np.histogram(samples, 100)
      hist_bin_centers = (hist_bin_edges[:-1] + hist_bin_edges[1:]) / 2
      largest_count = np.argmax(counts)
      anchored_pt = hist_bin_centers[largest_count]  # the mode of the dist
    else:
      counts, hist_bin_edges = np.histogramdd(samples, 100)
      hist_bin_centers = [(hist_bin_edges[x][:-1] + hist_bin_edges[x][1:]) / 2
                          for x in range(len(hist_bin_edges))]
      largest_count = np.unravel_index(np.argmax(counts), counts.shape)
      anchored_pt = np.array(
          [hist_bin_centers[coord_idx][largest_count[coord_idx]]
           for coord_idx in range(counts.ndim)])
      #^ the mode of the dist, in n dimensions
  elif placement_scheme == 'on_median':
    if samples.ndim == 1:
      anchored_pt = np.median(samples)
    else:
      # the geometric median is a high-dimensional generalization of the median.
      # It minimizes the sum of distances, NOT the sum of squared distances,
      # which makes it *different from the multvariate mean, or centroid*. You
      # can verify this for yourself on synthetic data.
      anchored_pt = np.array(hdmedians.geomedian(samples, axis=0))
  elif placement_scheme == 'on_mean':
    anchored_pt = np.mean(samples, axis=0)
  elif placement_scheme == 'on_zero':
    if samples.ndim == 1:
      anchored_pt = 0.0
    else:
      anchored_pt = np.zeros((samples.shape[1], ))
  else:
    raise KeyError('Unrecognized placement scheme ' + placement_scheme)

  max_val_each_dim = np.max(samples, axis=0)
  min_val_each_dim = np.min(samples, axis=0)
  assert np.all(anchored_pt < max_val_each_dim)
  assert np.all(anchored_pt >= min_val_each_dim)
  num_pts_lower = np.floor((anchored_pt - min_val_each_dim) / binwidth)
  num_pts_higher = np.floor((max_val_each_dim - anchored_pt) / binwidth)
  num_a_pts_each_dim = num_pts_lower + num_pts_higher + 1
  if samples.ndim == 1:
    assignment_pts = np.linspace(anchored_pt - num_pts_lower * binwidth,
                                 anchored_pt + num_pts_higher * binwidth,
                                 num_a_pts_each_dim)
    quantized_code, cluster_assignments = quantize(samples, assignment_pts, True)
  else:
    # Instead of taking a cartesian product of `n` scalar assignment-points,
    # pass them as is to quantize() routine. There, we quantize each of the
    # `n` dimensions individually using the corresponding scalar assignment-points.
    # Most fot he bins (in the cartesian product space) remain empty, discard them.
    # Recompute list of non-empty bins/vector assignment-points.

    assignment_pts = [np.linspace(anchored_pt[x] - num_pts_lower[x] * binwidth[x],
                      anchored_pt[x] + num_pts_higher[x] * binwidth[x],
                      num_a_pts_each_dim[x]) for x in range(samples.shape[1])]
    num_apts_orig = np.prod(np.array([len(x) for x in assignment_pts]))
    quantized_code, cluster_assignments = quantize(samples, assignment_pts, True)
    assignment_pts, cluster_assignments = np.unique(quantized_code, axis=0, return_inverse=True)
    num_apts_final = len(assignment_pts)
    print("Trimmed empty assignment points: {} -> {}".format(num_apts_orig, num_apts_final))

  if samples.ndim == 1:
    MSE = np.mean(np.square(quantized_code - samples))
  else:
    MSE = np.mean(np.sum(np.square(quantized_code - samples), axis=1))

  cword_probs = calculate_assignment_probabilites(cluster_assignments,
                                                  assignment_pts.shape[0])
  assert np.isclose(np.sum(cword_probs), 1.0)
  nonzero_prob_pts = np.where(cword_probs != 0)  # avoid log2(0)
  shannon_entropy = -1 * np.sum(
      cword_probs[nonzero_prob_pts] * np.log2(cword_probs[nonzero_prob_pts]))

  # Sort assignment_pts, cluster_assignments in decreasing order of probabliity
  sorted_i = np.flip(np.argsort(cword_probs),axis=0)
  cword_probs = cword_probs[sorted_i]
  assignment_pts = assignment_pts[sorted_i, :]
  cluster_assignments = np.argsort(sorted_i)[cluster_assignments]

  return assignment_pts, cluster_assignments, MSE, shannon_entropy


def quantize(raw_vals, assignment_vals, return_cluster_assignments=False):

  def nn(X, Y):
    """For each element in X, finds NN in Y when both X,Y are scalar arrays"""
    if len(Y)==1:
      # Everything gets assigned to this point
      return np.zeros((len(raw_vals),), dtype='int')
    else:
      # Here, we use sorted bin edges and the assignment complexity is
      # (I believe) logarithmic in the number of intervals.
      X_ind = np.argsort(X)
      Y_ind = np.argsort(Y)
      X_iind = np.argsort(X_ind)
      Y_iind = np.argsort(Y_ind)
      X_sorted = X[X_ind]
      Y_sorted = Y[Y_ind]
      edges = (Y_sorted[:-1] + Y_sorted[1:])/2
      neigh_XsYs = np.digitize(X_sorted, edges)
      neigh = Y_ind[neigh_XsYs[X_iind]]
      return neigh

  if raw_vals.ndim == 1:
    c_assignments = nn(raw_vals, assignment_vals)
    quantized_code = assignment_vals[c_assignments]
  else:
    # Quantize each dimension independently
    #   assignment_vals: (n,m)
    # Return
    #   quantized_code: (d, n)
    #   c_assignments: (d, n)
    c_assignments = np.zeros(raw_vals.shape)
    quantized_code = np.zeros(raw_vals.shape)
    for i in range(raw_vals.shape[1]):
      raw_vals_i = raw_vals[:,i]
      assignment_vals_i = assignment_vals[i]
      c_assignments_i = nn(raw_vals_i, assignment_vals_i)
      c_assignments[:,i] = c_assignments_i
      quantized_code[:,i] = assignment_vals_i[c_assignments_i]

  if return_cluster_assignments:
    return quantized_code, c_assignments
  else:
    return quantized_code

def calculate_assignment_probabilites(assignments, num_clusters):
  temp = np.arange(num_clusters)
  hist_b_edges = np.hstack([-np.inf, (temp[:-1] + temp[1:]) / 2, np.inf])
  assignment_counts, _ = np.histogram(assignments, hist_b_edges)
  empirical_density = assignment_counts / np.sum(assignment_counts)
  return empirical_density
