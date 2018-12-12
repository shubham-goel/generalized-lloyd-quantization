"""
An implementation of the 'optimal' generalized Lloyd algorithm

This is sometimes called 'Entropy Constrained Vector Quantization'. Instead
of trying to find assignment points that just minimize the mean-squared error
of the quantization, our fitting procedure also takes into account the codeword
lengths, via a lagrange multiplier which can be interpreted as (loosely)
enforcing an entropy constraint on the solution. This extra sensitivity of the
quantization to the actual entropy of the resulting code is what makes this
method 'optimal'.

This is currently implemented only for the 2-norm-squared error distortion
metric but is fully general for p-norms and the general quadratic form
e^T R e where e is the error vector. The computational or analytical details
of using these other distortion measures may be less desirable than for the
canonical 2-norm-squared distortion metric.

.. [1] Berger, T. (1982). Minimum entropy quantizers and permutation codes.
       IEEE transactions on information theory, 28(2), 149-157.

.. [2] Chou, P. A., Lookabaugh, T., & Gray, R. M. (1989). Entropy-constrained
       vector quantization. IEEE transactions on acoustics, speech, and signal
       processing, 37(1), 31-42.
"""
import copy
import numpy as np
import torch

def compute_quantization(samples, init_assignment_pts,
                         init_assignment_codeword_lengths,
                         lagrange_mult=1., epsilon=1e-5, device='cuda',
                         nn_method='brute_break'):
  """
  Implements so-called entropy constrained vector quantization (ECVQ)

  This takes the same basic setup as generalized Lloyd but instead we augment
  the traditional distance metric for computing the partition of the samples
  with a lagrange multiplier term that can be used to effectively constrain
  the total rate of the code assuming that we allocate codewords with
  different lengths to each assignment point. We'll call this our code cost.
  Our policy will be to eliminate assignment points with zero empirical
  probability so that the final number of assignment points may be smaller
  than what we start with.

  Parameters
  ----------
  samples : ndarray (d, n) or (d,)
      This is an array of d samples of an n-dimensional random variable
      that we wish to find the ECVQ quantizer for. If these are scalar random
      variables, we will accept a 1d array as input.
  init_assignment_pts : ndarray (m, n) or (m,)
      This is an array of some initial guesses for m total assignment points
      for the quantizer. We may prune this list in the optimization procedure
  init_assignment_codeword_lengths : ndarray (m,)
      The starting lengths for each assignment point. This will be changed in
      the first iteration to reflect the empirical probability of this codeword.
      Each of the components should be in the open interval (0, inf)
  lagrange_mult : float
      This is our knob to set the rate. We might have to sweep it carefully to
      trace out a finely sampled R/D curve.
  epsilon : float, optional
      The tolerance for change in code cost after which we decided we
      have converged. Default 1e-5.

  Returns
  -------
  assignment_pts : ndarray (m, n) or (m,)
      The converged assignment points
  cluster_assignments : ndarray (d, )
      For each sample, the index of the codeword to which optimal Lloyd
      quantization assigns this datapoint. We can compute the actual quantized
      values outside this function by invoking
      `assignment_pts[cluster_assignments]`
  MSE : float
      The mean squared error (the mean l2-normed-squared to be precise) for the
      returned quantization.
  shannon_entropy : float
      The (empirical) Shannon entropy for this code. We can say that assuming
      we use a lossless binary source code, that our expected codeword length
      is precisely this value
  """
  # get rid of the original references to make sure we don't modify the data
  # in the calling scope
  samples = torch.tensor(samples, dtype=torch.float, device=device)
  assignment_pts = torch.tensor(init_assignment_pts, dtype=torch.float, device=device)
  codeword_lengths = torch.tensor(init_assignment_codeword_lengths, dtype=torch.float, device=device)
  lagrange_mult = float(lagrange_mult)

  if samples.dim() == 1:
    assert(assignment_pts.ndim == 1)
    samples = samples[:,None]
    assignment_pts = assignment_pts[:,None]

  # Sanity Check
  assert(samples.dim() == 2)
  assert(assignment_pts.dim() == 2)
  assert(samples.shape[1] == assignment_pts.shape[1])
  assert(codeword_lengths.shape == (assignment_pts.shape[0],))
  assert(isinstance(lagrange_mult, float))
  assert(isinstance(epsilon, float))

  lagrange_mult = lagrange_mult * torch.mean(torch.std(samples, dim=0))
  #^ put effective lagrange mult on a sort of normalized scale
  #  with the standard deviation of our samples

  # partition the data into appropriate clusters
  quantized_code, cluster_assignments, assignment_pts, codeword_lengths = \
      partition_with_drops(samples, assignment_pts,
                           codeword_lengths, lagrange_mult, device=device,
                           nn_method=nn_method)

  MSE = torch.mean(torch.sum((quantized_code - samples).pow(2), dim=1))

  cword_probs = 2.**(-1 * codeword_lengths)
  shannon_entropy = torch.sum(cword_probs * codeword_lengths)
  code_cost = MSE + lagrange_mult * shannon_entropy

  while True:
    old_code_cost = code_cost
    # update the centroids based on the current partition
    for bin_idx in range(assignment_pts.shape[0]):
      binned_samples = samples[cluster_assignments == bin_idx]
      assert len(binned_samples) > 0
      assignment_pts[bin_idx] = torch.mean(binned_samples, dim=0)
      # the centroid rule doesn't change from unconstrained LBG

    # partition the data into appropriate clusters
    quantized_code, cluster_assignments, assignment_pts, codeword_lengths = \
        partition_with_drops(samples, assignment_pts,
                             codeword_lengths, lagrange_mult, device=device,
                             nn_method=nn_method)

    MSE = torch.mean(torch.sum((quantized_code - samples)**2, dim=1))

    cword_probs = 2.**(-1 * codeword_lengths)
    shannon_entropy = torch.sum(cword_probs * codeword_lengths)
    code_cost = MSE + lagrange_mult * shannon_entropy

    if not torch.isclose(old_code_cost, code_cost):
      assert code_cost <= old_code_cost, 'uh-oh, code cost increased'

    if old_code_cost == 0.0:  # avoid divide by zero below
      break
    if (torch.abs(old_code_cost - code_cost) / old_code_cost) < epsilon:
      break
      #^ this algorithm provably reduces this cost or leaves it unchanged at
      #  each iteration so the boundedness of this cost means this is a
      #  valid stopping criterion

  assignment_pts = assignment_pts.cpu().numpy()
  cluster_assignments = cluster_assignments.cpu().numpy()
  MSE = MSE.item()
  shannon_entropy = shannon_entropy.item()

  return assignment_pts, cluster_assignments, MSE, shannon_entropy


def quantize(raw_vals, assignment_vals, codeword_lengths,
             l_weight, return_cluster_assignments=False, device='cuda',
             nn_method='brute_break'):
  """
  Makes a quantization according to BOTH nearest neighbor and resulting code len

  We could assign the raw values to their nearest neighbor in assignment_vals,
  but that would ignore the resulting entropy of the assignment. Assuming that
  we use an optimal lossless code (Huffman, Arithmetic, etc.) for the
  assignments, the expected length of the code is arbitrarily close to the
  entropy. Instead, we will minimize a function which includes not only the
  distance to assignment points, but also has a lagrange multiplier term
  that accounts for the codeword length.

  Parameters
  ----------
  raw_vals : torch.tensor (d, n) or (d,)
      The raw values to be quantized according to the assignment points and lens
  assignment_vals : torch.tensor (m, n) or (m,)
      The allowable assignment values. Every raw value will be assigned one
      of these values instead.
  codeword_lengths : torch.tensor (m,)
      The expected lengths of the codewords for each of the assignment vals.
  l_weight : float
      A value for the lagrange multiplier used in the augmented distance we use
      to make the quantization
  return_cluster_assignments : bool, optional
      Our default behavior is to just return the actual quantized values
      (determined by the assingment points). If this parameter is true,
      also return the index of assigned point for each of the rows in
      raw_vals (this is the identifier of which codeword was used to quantize
      this datapoint). Default False.
  nn_method: str \in (brute_torch, brute_break)
      Specifies the method to compute nearest neighbour.
  """
  assert len(assignment_vals) == len(codeword_lengths)
  # I could not easily find an implementation of nearest neighbors that would
  # use a generalized cost function rather than the l2-norm-squared to assign
  # the partition. Therefore, we'll (for now) calculate the cost of assigning
  # each point to each interval and then just take the minimum.
  if raw_vals.dim() == 1:
    raw_vals = raw_vals[:, None]
    assignment_vals = assignment_vals[:, None]

  raw_vals_pad = torch.cat((raw_vals,
                            torch.zeros((raw_vals.shape[0],1),device=device)),
                            dim=1)

  assignment_vals_pad = torch.cat((assignment_vals,
                                    torch.sqrt(l_weight * codeword_lengths)[:,None]),
                                    dim=1)


  if nn_method == 'brute_break':
    # Answer independent of raw_values**2 term
    assignment_vals_pad_norm = torch.norm(assignment_vals_pad, dim=1)**2
    corr = torch.mm(raw_vals_pad, assignment_vals_pad.t())
    l2_distance = assignment_vals_pad_norm[None, :] - 2*corr
    c_assignments = torch.argmin(l2_distance, dim=1)
  elif nn_method == 'brute_torch':
    l2_distance = (raw_vals_pad[:,None,:] - assignment_vals_pad[None,:,:]).pow(2).sum(dim=2)
    c_assignments = torch.argmin(l2_distance, dim=1)

  if return_cluster_assignments:
    return assignment_vals[c_assignments], c_assignments
  else:
    return assignment_vals[c_assignments]


def partition_with_drops(raw_vals, a_vals, c_lengths, l_weight, device='cuda',
                          nn_method='brute_break'):
  """
  Partition the data according to the assignment values.

  This is just a wrapper on the quantize() function above which, following the
  advice of Chou et al. (1989), drops assignment points from the quantization
  whenever there are quantization bins with no data in them.

  Parameters
  ----------
  raw_vals : torch.tensor (d, n) or (d,)
      The raw values to be quantized according to the assignment points
  a_vals : torch.tensor (m, n) or (m,)
      The *initial* allowable assignment values. These may change according to
      whether quantizing based on these initial points results in empty bins.
  c_lengths : torch.tensor (m, )
      The (precomputed) codeword lengths for each of the assignment points.
      These will have been computed purely based on the empirical entropy of the
      quantized code from the previous iteration of the algorithm.
  l_weight : float
      The value of the lagrange multiplier in the augmented cost function
  """
  quant_code, c_assignments = quantize(raw_vals, a_vals,
                                       c_lengths, l_weight, True,
                                       device=device, nn_method=nn_method)

  cword_probs = calculate_assignment_probabilites(c_assignments,
                                                  a_vals.shape[0])
  if torch.any(cword_probs == 0):
    assert(cword_probs.dim() == 1)
    nonzero_prob_pts = (cword_probs != 0).nonzero().squeeze(1)
    # the indexes of c_assignments should reflect these dropped bins
    temp = -1 * torch.ones(a_vals.shape[0], device=device).long()
    temp[nonzero_prob_pts] = torch.arange(len(nonzero_prob_pts), device=device)
    c_assignments = temp[c_assignments]
    assert((c_assignments >=0).all())

    a_vals = a_vals[nonzero_prob_pts]
    cword_probs = cword_probs[nonzero_prob_pts]
  # update c_lengths so that the returned values reflect the current assignment
  c_lengths = -1 * torch.log2(cword_probs)

  return quant_code, c_assignments, a_vals, c_lengths


def calculate_assignment_probabilites(assignments, num_clusters):
  """
  Just counts the occurence of each assignment to get an empirical pdf estimate
  """

  assignment_counts = torch.bincount(assignments, minlength=num_clusters)
  empirical_density = assignment_counts.float()/assignment_counts.sum().float()

  return empirical_density
