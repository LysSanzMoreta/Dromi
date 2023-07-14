"""
=======================
2023: Lys Sanz Moreta
Dromi: Python package for parallel computation of similarity measures among vector-encoded sequences
=======================
"""
import functools
import itertools
import operator
import time,os,sys
import datetime
import numpy as np
import multiprocessing
from collections import namedtuple
SimilarityResults = namedtuple("SimilarityResults",["positional_weights","percent_identity_mean","cosine_similarity_mean","kmers_pid_similarity","kmers_cosine_similarity_mean"])

def cosine_similarity(a,b,correlation_matrix=False,parallel=False): #TODO: import from utils?
    """Calculates the cosine similarity between 2 arrays.
    :param numpy array a: (max_len,aa_types) or (num_seq,max_len, aa_types)
    :param numpy array b: (max_len,aa_types) or (num_seq,max_len, aa_types)
    :param bool:Calculate matrix correlation(as in numpy coorcoef)"""
    n_a = a.shape[0]
    n_b = b.shape[0]
    diff_sizes = False
    if n_a != n_b:
        dummy_row = np.zeros((np.abs(n_a-n_b),) + a.shape[1:])
        diff_sizes = True
        if n_a < n_b:
            a = np.concatenate((a,dummy_row),axis=0)
        else:
            b = np.concatenate((b,dummy_row),axis=0)
    if np.ndim(a) == 1:
        num = np.dot(a,b)
        p1 = np.sqrt(np.sum(a**2)) #equivalent to p1 = np.linalg.norm(a)
        p2 = np.sqrt(np.sum(b**2))
        cosine_sim = num/(p1*p2)
        return cosine_sim

    elif np.ndim(a) == 2:
        if correlation_matrix:
            b = b - b.mean(axis=1)[:, None]
            a = a - a.mean(axis=1)[:, None]

        num = np.dot(a, b.T) #[seq_len,21]@[21,seq_len] = [seq_len,seq_len]
        p1 =np.sqrt(np.sum(a**2,axis=1))[:,None] #[seq_len,1]
        p2 = np.sqrt(np.sum(b ** 2, axis=1))[None, :] #[1,seq_len]
        #print(p1*p2)
        cosine_sim = num / (p1 * p2)
        if parallel:
            return cosine_sim[None,:]
        else:
            return cosine_sim
    else: #TODO: use elipsis for general approach?
        if correlation_matrix:
            b = b - b.mean(axis=2)[:, :, None]
            a = a - a.mean(axis=2)[:, :, None]
        num = np.matmul(a[:, None], np.transpose(b, (0, 2, 1))[None,:]) #[n,n,seq_len,seq_len]
        p1 = np.sqrt(np.sum(a ** 2, axis=2))[:, :, None] #Equivalent to np.linalg.norm(a,axis=2)[:,:,None]
        p2 = np.sqrt(np.sum(b ** 2, axis=2))[:, None, :] #Equivalent to np.linalg.norm(b,axis=2)[:,None,:]
        cosine_sim = num / (p1[:,None]*p2[None,:])

        if diff_sizes: #remove the dummy creation that was made avoid shape conflicts
            remove = np.abs(n_a-n_b)
            if n_a < n_b:
                cosine_sim = cosine_sim[:-remove]
            else:
                cosine_sim = cosine_sim[:,:-remove]

        return cosine_sim

def extract_windows_vectorized(array, clearing_time_index, max_time, sub_window_size,only_windows=True): #TODO: import from utils
    """
    Creates indexes to extract kmers from a sequence, such as:
         seq =  [A,T,R,P,V,L]
         kmers_idx = [0,1,2,1,2,3,2,3,4,3,4,5]
         seq[kmers_idx] = [A,T,R,T,R,P,R,V,L,P,V,L]
    From https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
    :param int clearing_time_index: Indicates the starting index (0-python idx == 1 clearing_time_index;-1-python idx == 0 clearing_time_index)
    :param max_time: max sequence len
    :param sub_window_size:kmer size
    """
    start = clearing_time_index + 1 - sub_window_size + 1
    sub_windows = (
            start +
            # expand_dims are used to convert a 1D array to 2D array.
            np.arange(sub_window_size)[None,:]  + #[0,1,2] ---> [[0,1,2]]
            np.arange(max_time + 1)[None,:].T  #[0,...,max_len+1] ---expand dim ---> [[[0,...,max_len+1] ]], indicates the
    ) # The first row is the sum of the first row of a + the first element of b, and so on (in the diagonal the result of a[None,:] + b[None,:] is placed (without transposing b). )

    if only_windows:
        return sub_windows
    else:
        return array[:,sub_windows]

class KmersFilling(object):
    """Fills in the cosine similarities of the overlapping kmers (N,nkmers,ksize) onto (N,max_len)"""
    def __init__(self,rows_idx_a,rows_idx_b,cols_idx_a,cols_idx_b):
        self.rows_idx_a = rows_idx_a
        self.rows_idx_b = rows_idx_b
        self.cols_idx_a = cols_idx_a
        self.cols_idx_b = cols_idx_b
        self.iterables = self.rows_idx_a,self.rows_idx_b,self.cols_idx_a,self.cols_idx_b
    def run(self,weights,hotspots):
        return list(map(lambda row_a,row_b,col_a,col_b: self.fill_kmers_array(row_a,row_b,col_a,col_b,weights,hotspots),self.rows_idx_a,self.rows_idx_b,self.cols_idx_a,self.cols_idx_b))
    def run_pool(self,pool,weights,hotspots): #TODO: Does not seem the speed bottleneck, but could be better
        fixed_args = weights,hotspots
        return list(pool.map(self.fill_kmers_array, list(zip(zip(*self.iterables), itertools.repeat(fixed_args)))))
    def fill_kmers_array(self,row_a,row_b,col_a,col_b,a, b):

        a = a .copy()
        a[row_a, col_a[0]:col_a[1]] += b[row_b,col_b]
        return a

class MaskedMeanParallel(object):
   """Performs parallel computation of masked (ignoring paddings) mean to calculate the positional weights (similarity/conservation) measure"""
   def __init__(self,iterables,fixed_args,kmers =False):
        self.fixed = fixed_args
        self.kmers = kmers
        self.splits = iterables["splits"]
        self.diag_idx_1 = iterables["diag_idx_1"]
        if self.kmers:
            #self.splits_mask = iterables["splits_mask"]
            self.kmers_idxs = iterables["kmers_idxs"]
            self.iterables = self.splits,self.kmers_idxs,self.diag_idx_1
        else:
            self.positional_idxs = iterables["positional_idxs"]
            self.iterables = self.splits, self.positional_idxs, self.diag_idx_1
   def run(self, pool):
       if self.kmers:
            return list(pool.map(masked_mean_loop_kmers, list(zip(zip(*self.iterables), itertools.repeat(self.fixed)))))
       else:
            return list(pool.map(masked_mean_loop, list(zip(zip(*self.iterables), itertools.repeat(self.fixed)))))

def masked_mean_loop_kmers(params):
    """Helper function for Multiprocessing
    :param params : zipped list of 2 types of parameters"""
    iterables, fixed = params
    return calculate_masked_mean_kmers(iterables,fixed_args=fixed)

def masked_mean_loop(params):
    """Helper function for Multiprocessing
    :param params : zipped list of 2 types of parameters"""
    iterables, fixed = params
    return calculate_masked_mean(iterables,fixed_args=fixed)

def calculate_masked_mean_kmers(iterables_args,fixed_args): #TODO: Needs to be reviewed or deleted
    """Calculates the average cosine similarity of each sequence to the 1 or 3 neighbouring elements of every other sequence & ignoring paddings
    :param iterables_args: Variable arguments
    :param fixed_args: Inmmutable/static arguments"""
    nkmers,kmers_mask= fixed_args #kmers_mask = [N,nkmers,ksize]
    hotspots,kmer_idx,diag_idx_1 = iterables_args #hotspots = [batch_size,N,nkmers,nkmers,ksize)
    print("--------------{}-------------".format(kmer_idx))
    diag_idx_0 = np.arange(0,hotspots.shape[0]) #in case there are uneven splits
    hotspots[diag_idx_0,diag_idx_1] = 0 #ignore self cosine similarity
    #hotspots = hotspots[seq_idx]
    hotspots_mask = np.zeros_like(hotspots)
    if kmer_idx -1 < 0:
        neighbour_kmers_idx = np.array([kmer_idx,kmer_idx +1,kmer_idx+2])
    elif kmer_idx + 1 == nkmers:
        neighbour_kmers_idx = np.array([kmer_idx-2,kmer_idx-1,kmer_idx])
    else:
        neighbour_kmers_idx = np.array([kmer_idx-1,kmer_idx,kmer_idx +1])
    hotspots_mask[:,:,:][:,:,:,neighbour_kmers_idx] = 1 #True (use for calculation)
    hotspots_mask = hotspots_mask.astype(bool)
    #Highlight: refine the mask to ignore also the paddings
    #TODO: REVIEW AGAIN!!! Investigate: #neighbour_kmers_idx = np.array(kmer_idx)

    kmers_mask_0 = kmers_mask[diag_idx_1] #(0 and 1 are switched on purpose, it is not an accident), select the mask of the sequences in the batch [batch_size, nkmers,,ksize]
    kmers_mask_0 = np.repeat(kmers_mask_0[:, :,None], nkmers, axis=2)
    kmers_mask_0 = kmers_mask_0.transpose((0,2,1,3))
    kmers_mask = np.repeat(kmers_mask[:, :, None], nkmers, axis=2)
    kmers_mask_split = (kmers_mask_0[:, None] * kmers_mask[None, :]).astype(bool)
    hotspots_mask *= kmers_mask_split
    hotspots_masked_mean = np.ma.masked_array(hotspots, mask=~hotspots_mask, fill_value=0.).mean(1)  # Highlight: In the mask if True means to mask and ignore!!!!


    hotspots_masked_mean = np.ma.masked_array(hotspots_masked_mean, mask=~kmers_mask_0, fill_value=0.).mean(2) #TODO: Should it be mean 3?
    return np.ma.getdata(hotspots_masked_mean) #[batch_size,nkmers,ksize]

def importance_weight_kmers(hotspots,nkmers,ksize,max_len,positional_mask,overlapping_kmers,batch_size): #TODO: Needs to be reviewed or deleted
    """Weighting cosine similarities across kmers to find which positions in the sequence are more conserved
    WARNING: Experimental, use without guarantees
    :param hotspots  = kmers_matrix_cosine_diag_ij : [N,N,nkmers,nkmers,ksize]
    :param int nkmers: Number of kmers in the sequence
    :param int ksize: Size of the kmer
    :param int max_len: Length of the longest sequence in the dataset
    :param array positional_mask: Array indicating the sequence paddings
    :param array overlapping_kmers: Array containing the sequences subdivided in kmers
    :param int batch_size: Size of the sequence subset
    """
    print("Calculating positional importance weights based on neighbouring-kmer cosine similarity")
    n_seqs = hotspots.shape[0]
    #Highlight: Masked mean only over neighbouring kmers
    split_size = [int(hotspots.shape[0] / batch_size) if not batch_size > hotspots.shape[0] else 1][0]
    splits = np.array_split(hotspots, split_size)
    splits = [[split]*nkmers for split in splits]
    splits = functools.reduce(operator.iconcat, splits, [])
    kmers_mask = positional_mask[:,overlapping_kmers]
    # splits_mask = np.array_split(kmers_mask, split_size)
    # splits_mask = [[split]*nkmers for split in splits_mask]
    # splits_mask = functools.reduce(operator.iconcat, splits_mask, [])
    kmers_idx = np.tile(np.arange(0, nkmers), len(splits))
    diag_idx = np.diag_indices(n_seqs)
    #diag_idx_0 = diag_idx[0][:batch_size]
    diag_idx_1 = [[i]*nkmers for i in np.array_split(diag_idx[1],split_size)]
    diag_idx_1 = functools.reduce(operator.iconcat, diag_idx_1, [])
    fixed_args = nkmers,kmers_mask
    # for s,k_i,diag_1 in zip(splits,kmers_idx,diag_idx_1):
    #     k_i = 8
    #     iterables_args = s,k_i,diag_1
    #     r = calculate_masked_mean(iterables_args,fixed_args)
    #     exit()
    # exit()
    args_iterables = {"splits":splits,
                      "kmers_idxs": kmers_idx,
                      "diag_idx_1":diag_idx_1}
    args_fixed = nkmers,kmers_mask
    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
        results = MaskedMeanParallel(args_iterables,args_fixed).run(pool)
        #zipped_results =list(zip(*results))
        results = [sum(results[x:x+nkmers]) for x in range(0, len(results), nkmers)] #divide again by nkmers
    hotspots_mean = np.concatenate(results,axis=0)
    print("Done calculating the masked average")

    positions_weights = np.zeros((n_seqs, max_len))
    rows_idx_a = np.repeat(np.arange(0, n_seqs), nkmers) #select rows from weights dataframe
    rows_idx_b = np.repeat(np.arange(0, n_seqs), nkmers) #select rows from hotspots dataframe
    cols_idx_b = np.tile(np.arange(0, nkmers), n_seqs)  # [0,1,0,1,0,1,...] --> select from the hotspots mean dataframe
    cols_idx_a_0 = np.tile(np.arange(0, nkmers), n_seqs) #[0,1,2,3,...]
    cols_idx_a_1 = np.tile(np.arange(ksize, nkmers +ksize), n_seqs) #[3,4,5,6,]
    cols_idx_a = np.concatenate([cols_idx_a_0[:, None], cols_idx_a_1[:, None]], axis=1)

    left_divisors,right_divisors = [1,2], [2,1]
    if max_len == ksize:
        divisors = np.ones(max_len)
    elif max_len<5:
        divisors = left_divisors + right_divisors
    else:
        divisors = left_divisors  + (max_len -4)*[ksize] + right_divisors

    divisors = np.array(divisors)
    positional_weights = sum(KmersFilling(rows_idx_a, rows_idx_b, cols_idx_a,cols_idx_b).run(positions_weights,hotspots_mean))
    positional_weights /= divisors
    positional_weights = (positional_weights - positional_weights.min()) / (positional_weights.max() - positional_weights.min()) #min max scale
    positional_weights*= positional_mask
    print("Finished positional weights")
    return positional_weights

def calculate_masked_mean(iterables_args,fixed_args):
    """Calculates the average cosine similarity of each sequence to the 1 or 3 neighbouring elements of every other sequence & ignoring paddings
    :param iterables_args: Variable arguments
    :param fixed_args: Inmmutable/static arguments"""
    max_len,positional_mask,neighbours= fixed_args #kmers_mask = [N,nkmers,ksize]
    hotspots,positional_idx,diag_idx_1 = iterables_args #hotspots = [batch_size,N,max_len,max_len)
    print("-----------positional idx: {}-------------".format(positional_idx))
    diag_idx_0 = np.arange(0,hotspots.shape[0]) #in case there are uneven splits
    hotspots[diag_idx_0,diag_idx_1] = 0 #ignore self cosine similarity (
    positional_weights = np.zeros((hotspots.shape[0],max_len))
    #hotspots = hotspots[seq_idx]
    hotspots_mask = np.zeros_like(hotspots)
    if positional_idx -1 < 0:
        if neighbours == 1:
            neighbour_positions_idx = np.array([positional_idx])
            divisor = 1
        else:
            neighbour_positions_idx = np.array([positional_idx,positional_idx +1,positional_idx+2])
            divisor = 3

    elif positional_idx + 1 == max_len or positional_idx == 8 or positional_idx == 9:
        if neighbours == 1:
            neighbour_positions_idx = np.array([positional_idx])
            divisor = 1
        else:
            neighbour_positions_idx = np.array([positional_idx-2,positional_idx-1,positional_idx])
            divisor = 3

    else:
        if neighbours == 1:
            neighbour_positions_idx = np.array([positional_idx])
            divisor = 1
        else:
            neighbour_positions_idx = np.array([positional_idx-1,positional_idx,positional_idx +1])
            divisor = 3

    hotspots_mask[:,:,positional_idx][:,:,neighbour_positions_idx] = 1 #True (use for calculation)
    hotspots_mask = hotspots_mask.astype(bool)
    batch_mask = positional_mask[diag_idx_1] #(0 and 1 are switched on purpose, it is not an accident), select the mask of the sequences in the batch [batch_size, nkmers,,ksize]
    batch_mask_expanded = np.repeat(batch_mask[:, :, None], max_len, axis=2)
    batch_mask_expanded = np.repeat(batch_mask_expanded[:, None, :], positional_mask.shape[0], axis=1)
    positional_mask_expanded = np.repeat(positional_mask[:, :, None], max_len, axis=2)
    positional_mask_expanded = np.repeat(positional_mask_expanded[None, :], batch_mask.shape[0], axis=0)
    positional_mask_expanded = positional_mask_expanded * batch_mask_expanded.transpose((0, 1, 3, 2))
    hotspots_mask *= positional_mask_expanded
    hotspots_masked_mean = ((hotspots*hotspots_mask.astype(int)).sum(-1))/divisor
    hotspots_masked_mean = np.sum(hotspots_masked_mean.sum(-1),-1)/(hotspots.shape[1]-1) #.mean(-1) #sum first, then mean accross all sequences (minus 1(itself)), because the other positions are 0
    positional_weights[:,positional_idx] = hotspots_masked_mean

    return positional_weights #[batch_size,max_len]

def importance_weight(hotspots,max_len,positional_mask,batch_size,neighbours):
    """Weighting cosine similarities across neighbouring aminoacids to find which positions in the sequence are more conserved
    :param hotspots  = kmers_matrix_cosine_diag_ij : [N,N,nkmers,nkmers,ksize]
    :param hotspots  = kmers_matrix_cosine_diag_ij : [N,N,nkmers,nkmers,ksize]
    :param int max_len: Length of the longest sequence in the dataset
    :param array positional_mask: Array indicating the sequence paddings
    :param int batch_size: Size of the sequence subset
    """
    print("Calculating positional importance weights based on neighbouring-aminoacids cosine similarity")

    n_seqs = hotspots.shape[0]
    #Highlight: Masked mean only over neighbouring kmers
    split_size = [int(hotspots.shape[0] / batch_size) if not batch_size > hotspots.shape[0] else 1][0]
    splits = np.array_split(hotspots, split_size)
    splits = [[split]*max_len for split in splits]
    splits = functools.reduce(operator.iconcat, splits, [])
    positional_idxs = np.tile(np.arange(0, max_len), split_size)

    diag_idx = np.diag_indices(n_seqs)
    diag_idx_1 = [[i]*max_len for i in np.array_split(diag_idx[1],split_size)]
    diag_idx_1 = functools.reduce(operator.iconcat, diag_idx_1, [])

    # fixed_args = max_len,positional_mask,neighbours
    # for s,p_i,diag_1 in zip(splits,positional_idxs,diag_idx_1):
    #     p_i=1
    #     iterables_args = s,p_i,diag_1
    #     r = calculate_masked_mean(iterables_args,fixed_args)
    # exit()

    args_iterables = {"splits":splits,
                      "positional_idxs": positional_idxs,
                      "diag_idx_1":diag_idx_1}
    args_fixed = max_len,positional_mask,neighbours
    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
        results = MaskedMeanParallel(args_iterables,args_fixed).run(pool)
        #zipped_results =list(zip(*results))
        results = [sum(results[x:x+max_len]) for x in range(0, len(results), max_len)] #split again by max len
    positional_weights = np.concatenate(results,axis=0) #Highlight: divide by the number of neighbours used to compute the mean TODO: Make a range of neighbour positions to use?


    print("Done calculating the masked average")
    #positional_weights = (positional_weights - positional_weights.min()) / (positional_weights.max() - positional_weights.min()) #min max scale
    positional_weights*= positional_mask
    print("Finished positional weights")
    return positional_weights

def process_value(iterables_args,fixed_args):
    """Computes similarities metrics (pairwise identity, cosine similarity ... ) among a set of arrays"""

    i,j,shift,start_store_point,end_store_point,store_point_helper,start_store_point_i,end_store_point_i = iterables_args
    splits, mask_splits, n_data,max_len, overlapping_kmers, diag_idx_ksize, diag_idx_maxlen, diag_idx_nkmers = fixed_args
    print(" ------------  i: {}----------------------------".format(i))
    curr_array = splits[i]
    curr_mask = mask_splits[i]
    n_data_curr = curr_array.shape[0]
    rest_splits = splits.copy()[shift:]
    # Highlight: Define intermediate storing arrays #TODO: They can be even smaller to have shape sum(rest_splits.shape)
    start_i = time.time()
    print("###### j {} ##########################".format(j))
    r_j = rest_splits[j] #next array
    r_j_mask = mask_splits[j + shift]
    cosine_sim_j = cosine_similarity(curr_array, r_j, correlation_matrix=False)
    if np.ndim(curr_array) == 2:  # Integer encoded #TODO: Delete and force to have dimensions [N,L,1]
        pairwise_sim_j = (curr_array[None, :] == r_j[:, None]).astype(int)
        pairwise_matrix_j = (curr_array[:, None, :, None] == r_j[None, :, None, :]).astype(int)
    else:
        pairwise_sim_j = (curr_array[:, None] == r_j[None, :]).all((-1)).astype(int)  # .all((-2,-1)) #[1,L]
        pairwise_matrix_j = (curr_array[:, None, :, None] == r_j[None, :, None, :]).all((-1)).astype(float)  # .all((-2,-1)) #[1,L,L]
    # Highlight: Create masks to ignore the paddings of the sequences
    kmers_mask_curr_i = curr_mask[:, overlapping_kmers]
    kmers_mask_r_j = r_j_mask[:, overlapping_kmers]
    kmers_mask_ij = (kmers_mask_curr_i[:, None] * kmers_mask_r_j[None, :]).mean(-1)
    kmers_mask_ij[kmers_mask_ij != 1.] = 0.
    kmers_mask_ij = kmers_mask_ij.astype(bool)
    pid_mask_ij = curr_mask[:, None] * r_j_mask[None, :]
    # Highlight: Further transformations: Basically slice the overlapping kmers and organize them to have shape
    #  [m,n,kmers,nkmers,ksize,ksize], where the diagonal contains the pairwise values between the kmers
    kmers_matrix_pid_ij = pairwise_matrix_j[:, :, :, overlapping_kmers][:, :, overlapping_kmers].transpose(0, 1,
                                                                                                           4, 2,
                                                                                                           3, 5)
    kmers_matrix_cosine_ij = cosine_sim_j[:, :, :, overlapping_kmers][:, :, overlapping_kmers].transpose(0, 1,
                                                                                                         4, 2,
                                                                                                         3, 5)
    # Highlight: Apply masks to calculate the similarities. NOTE: To get the data with the filled value use k = np.ma.getdata(kmers_matrix_diag_masked)
    ##PERCENT IDENTITY (all vs all comparison)
    #Highlight: Prepare the mask according to the mask of the 2 compared arrays
    curr_mask_expanded = np.repeat(curr_mask[:, :, None], max_len, axis=2)
    curr_mask_expanded = np.repeat(curr_mask_expanded[:, None, :], r_j_mask.shape[0], axis=1)
    r_j_mask_expanded = np.repeat(r_j_mask[:, :, None], max_len, axis=2)
    r_j_mask_expanded = np.repeat(r_j_mask_expanded[None, :], curr_mask.shape[0], axis=0)
    matrix_mask_ij = curr_mask_expanded * r_j_mask_expanded.transpose((0, 1, 3, 2))
    ##PERCENT IDENTITY (binary pairwise comparison) ###############
    pid_pairwise_matrix_ij = np.ma.masked_array(pairwise_matrix_j, mask=~matrix_mask_ij, fill_value=0.) #[1,L,L] #TODO: Discard?
    percent_identity_mean_ij = np.ma.masked_array(pairwise_sim_j, mask=~pid_mask_ij, fill_value=0.).mean(-1)  # Highlight: In the mask if True means to mask and ignore!!!!
    ##COSINE SIMILARITY (all vs all cosine simlarity)########################
    cosine_sim_pairwise_matrix_ij = np.ma.masked_array(cosine_sim_j, mask=~matrix_mask_ij, fill_value=0.) # [1,L,L] # Highlight: In the mask if True means to mask and ignore!!!!
    ##COSINE SIMILARITY (pairwise comparison of cosine similarities_old)########################
    cosine_similarity_mean_ij = np.ma.masked_array(cosine_sim_j[:, :, diag_idx_maxlen[0], diag_idx_maxlen[1]],mask=~pid_mask_ij, fill_value=0.).mean(-1)  # Highlight: In the mask if True means to mask and ignore!!!!
    # KMERS PERCENT IDENTITY ############
    kmers_matrix_pid_diag_ij = kmers_matrix_pid_ij[:, :, :, :, diag_idx_ksize[0],diag_idx_ksize[1]]  # does not seem expensive
    kmers_matrix_pid_diag_mean_ij = np.mean(kmers_matrix_pid_diag_ij, axis=4)[:, :, diag_idx_nkmers[0],diag_idx_nkmers[1]]  # if we mask this only it should be fine
    kmers_pid_similarity_ij = np.ma.masked_array(kmers_matrix_pid_diag_mean_ij, mask=~kmers_mask_ij,fill_value=0.).mean(axis=2)
    # KMERS COSINE SIMILARITY ########################
    kmers_matrix_cosine_diag_ij = kmers_matrix_cosine_ij[:, :, :, :, diag_idx_ksize[0],diag_idx_ksize[1]]  # does not seem expensive
    kmers_matrix_cosine_diag_mean_ij = np.nanmean(kmers_matrix_cosine_diag_ij, axis=4)[:, :, diag_idx_nkmers[0],diag_idx_nkmers[1]]
    kmers_cosine_similarity_ij = np.ma.masked_array(kmers_matrix_cosine_diag_mean_ij, mask=~kmers_mask_ij,fill_value=0.).mean(axis=2)
    if i == j:# Highlight: When comparing an array to itself, round to nearest integer the diagonal values, due to precision issues, sometimes it computes 0.999999999 or 1.00000002 instead of 1.
        # Faster method that unravels the 2D array to 1D. Equivalent to: kmers_cosine_similarity_ij[np.diag_indices_from(cosine_similarity_mean_ij)] = np.rint(np.diagonal(kmers_cosine_similarity_ij))
        kmers_cosine_similarity_ij.ravel()[:kmers_cosine_similarity_ij.shape[1] ** 2:kmers_cosine_similarity_ij.shape[1] + 1] = np.rint(kmers_cosine_similarity_ij.ravel()[:kmers_cosine_similarity_ij.shape[1] ** 2:kmers_cosine_similarity_ij.shape[1] + 1])
        # Faster method that unravels the 2D array to 1D. Equivalent to: cosine_similarity_mean_ij[np.diag_indices_from(cosine_similarity_mean_ij)] = np.rint(np.diagonal(cosine_similarity_mean_ij))
        cosine_similarity_mean_ij.ravel()[:cosine_similarity_mean_ij.shape[1] ** 2:cosine_similarity_mean_ij.shape[1] + 1] = np.rint(cosine_similarity_mean_ij.ravel()[:cosine_similarity_mean_ij.shape[1] ** 2:cosine_similarity_mean_ij.shape[1] + 1])
    end_i = time.time()
    print("Time for finishing loop (i vs j) {}".format(str(datetime.timedelta(seconds=end_i - start_i))))

    return \
        cosine_sim_pairwise_matrix_ij,\
        percent_identity_mean_ij,\
        cosine_similarity_mean_ij,\
        kmers_cosine_similarity_ij,\
        kmers_pid_similarity_ij, \
        kmers_matrix_cosine_diag_ij,\
        start_store_point,end_store_point,start_store_point_i,end_store_point_i

def inner_loop(params):
    """Auxiliary function to SimilarityParallel"""
    iterables, fixed = params
    return process_value(iterables,fixed_args=fixed)

class SimilarityParallel:
   def __init__(self,iterables,fixed_args):
       self.fixed = fixed_args
       self.i_idx = iterables["i_idx"]
       self.j_idx = iterables["j_idx"]
       self.shifts = iterables["shifts"]
       self.start_store_points = iterables["start_store_points"]
       self.end_store_points = iterables["end_store_points"]
       self.store_point_helpers = iterables["store_point_helpers"]
       self.end_store_points_i = iterables["end_store_points_i"]
       self.start_store_points_i = iterables["start_store_points_i"]
       self.iterables = self.i_idx,self.j_idx,self.shifts,self.start_store_points,self.end_store_points,self.store_point_helpers,self.start_store_points_i,self.end_store_points_i

   def outer_loop(self, pool):
       return list(pool.map(inner_loop, list(zip(zip(*self.iterables), itertools.repeat(self.fixed)))))

def fill_array(array_fixed,ij,start,end,start_i,end_i):
    """Fill batch result in the corresponding slot
    :param array_fixed: Empty array
    :param ij: Pre-computed result (array) that will fill it the correct slot in the array_fixed
    :param int start: Indicates the row-wise start position where this batch is allocated
    :param int end: Indicates the row-wise end position where this batch is allocated
    :param int start_i: Indicates the column-wise start position where this batch is allocated
    :param int end_i: Indicates the column-wise end position where this batch is allocated
    """
    array_fixed[start:end,start_i:end_i] = ij
    return array_fixed

def fill_array_map(array_fixed,ij_arrays,starts,ends,starts_i,ends_i):
     """Fills the empty matrix with the results of each batch
    :param array_fixed: Empty array to fill in
    :param ij_arrays: Pre-computed result that will fill it the correct slot in the array_fixed
    :param int start: Indicates the row-wise start position where this batch is allocated
    :param int end: Indicates the row-wise end position where this batch is allocated
    :param int start_i: Indicates the column-wise start position where this batch is allocated
    :param int end_i: Indicates the column-wise end position where this batch is allocated
     """
     results = list(map(lambda ij,start,end,start_i,end_i: fill_array(array_fixed,ij,start,end,start_i,end_i),ij_arrays,starts,ends,starts_i,ends_i))
     return results[0]

def calculate_similarities(array, max_len, array_mask, storage_folder,batch_size=50, ksize=3,neighbours=1):
    """Batched method to calculate the cosine similarity and percent identity/pairwise distance between the (vector) encoded sequences.
    :param ndarray array: Numpy array of Vector encoded sequences [N,max_len,vector_dim], where N is the number of sequences in the array and vector_dim is the dimension if the encodings
    :param int max_len: Longest sequence in the array
    :param ndarray array_mask: Mask to indicate the paddings, where False means padding, True otherwise
    :param str storage_folder: Path to folder where to store the results
    :param batch_size: Number of sequences per batch, it automatically corrects for uneven splits
    :param int ksize: Kmer size
    :param neighbours: 1 or 3, this determines among how many sites/columns in the sequence the positional weights (conservation) are calculated.
           If it is set to 1 then the similarity will only computed among the current elements and the rest of elements in the same column.
           If set to 3 then it will be computed among the current element and the rest of elements in the same column, the left column and the right column.

    NOTE: Use smaller batches for faster results ( obviously to certain extent, check into balancing the batch size and the number of for loops)
    returns: namedtuple with the following outputs
        positional_weights = (n_data,max_len): Weights or residue conservation per site/column
        percent_identity_mean = (n_data,n_data) : 1 means the two aa sequences are identical, 0 completely different
        cosine_similarity_mean = (n_data,n_data):  1 means the two aa sequences are identical, -1 very disimilar
        kmers_pid_similarity = (n_data,n_data,nkmers)
        kmers_cosine_similarity = (n_data,n_data)
    """

    n_data = array.shape[0]
    if array.size == 0:
        print("Empty array")
    else:
        assert array_mask.shape == (n_data,max_len), "Your dataset mask has dimensions {}, while max_len is {}".format(array_mask.shape,max_len)
        assert batch_size <= n_data, "Please select a smaller batch size, current is {}. while dataset size is {}".format(batch_size,n_data)
        assert array.ndim == 3, "Please encode your sequences with ndim =3 ,such that if you have a 1 dimensional vector they will be encoded as [N,max_len,1]"
        split_size = [int(array.shape[0] / batch_size) if not batch_size > array.shape[0] else 1][0]
        splits = np.array_split(array, split_size)
        mask_splits = np.array_split(array_mask, split_size)
        print("Generated {} splits from {} data points".format(len(splits), n_data))

        if ksize >= max_len:
            ksize = max_len
        overlapping_kmers = extract_windows_vectorized(splits[0], 1, max_len - ksize, ksize, only_windows=True)

        #diag_idx_ndata =np.diag_indices(n_data)
        diag_idx_ksize = np.diag_indices(ksize)
        nkmers = overlapping_kmers.shape[0]
        diag_idx_nkmers = np.diag_indices(nkmers)
        diag_idx_maxlen = np.diag_indices(max_len)

        # Highlight: Initialize the storing matrices (in the future perhaps dictionaries? but seems to withstand quite a bit)
        #TODO: https://pythonspeed.com/articles/numpy-memory-footprint/
        percent_identity_mean = np.zeros((n_data, n_data),dtype=np.uint16)
        #pid_pairwise_matrix= np.zeros((n_data, n_data,max_len,max_len))
        cosine_similarity_mean = np.zeros((n_data, n_data),dtype=np.uint16)
        cosine_sim_pairwise_matrix= np.zeros((n_data, n_data,max_len,max_len),dtype=np.uint16)
        kmers_pid_similarity = np.zeros((n_data, n_data),dtype=np.uint16)
        kmers_cosine_similarity = np.zeros((n_data, n_data),dtype=np.uint16)
        #kmers_cosine_similarity_matrix_diag = np.zeros((n_data, n_data,nkmers,nkmers,ksize),dtype=np.uint16) #TODO: Reduce memory consumption
        #Highlight: Initialize the list of indexes for parallel computation
        # Iterables
        idx = list(range(len(splits)))
        shifts = []
        start_store_points = []
        start_store_points_i = []
        store_point_helpers = []
        end_store_points = []
        end_store_points_i = []
        i_idx = []
        j_idx = []
        start_store_point = 0
        store_point_helper = 0
        end_store_point = splits[0].shape[0]
        for i in idx:
            shift = i
            rest_splits = splits.copy()[shift:]
            start_store_point_i = 0 + store_point_helper
            end_store_point_i = rest_splits[0].shape[0] + store_point_helper  # initialize
            for j, r_j in enumerate(
                    rest_splits):  # calculate distance among all kmers per sequence in the block (n, n_kmers,n_kmers)
                i_idx.append(i)
                shifts.append(shift)
                j_idx.append(j)
                start_store_points.append(start_store_point)
                store_point_helpers.append(store_point_helper)
                end_store_points.append(end_store_point)
                start_store_points_i.append(start_store_point_i)
                end_store_points_i.append(end_store_point_i)
                start_store_point_i = end_store_point_i  # + store_point_helper
                if j + 1 < len(rest_splits):
                    end_store_point_i += rest_splits[j + 1].shape[0]  # + store_point_helper# it has to be the next r_j
            start_store_point = end_store_point
            if i + 1 < len(splits):
                store_point_helper += splits[i + 1].shape[0]
            if i + 1 != len(splits):
                end_store_point += splits[i + 1].shape[0]  # it has to be the next curr_array
            else:
                pass

        start = time.time()
        args_fixed = splits, mask_splits, n_data,max_len, overlapping_kmers, diag_idx_ksize, diag_idx_maxlen, diag_idx_nkmers
        args_iterables = {"i_idx":i_idx,
                          "j_idx":j_idx,
                          "shifts":shifts,
                          "start_store_points":start_store_points,
                          "start_store_points_i": start_store_points_i,
                          "store_point_helpers":store_point_helpers,
                          "end_store_points":end_store_points,
                          "end_store_points_i": end_store_points_i
                          }


        with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
            results = SimilarityParallel(args_iterables,args_fixed).outer_loop(pool)
            zipped_results =list(zip(*results))
            starts,ends,starts_i,ends_i = zipped_results[6],zipped_results[7],zipped_results[8],zipped_results[9]
            cosine_sim_pairwise_matrix_ij = zipped_results[0]
            cosine_sim_pairwise_matrix= fill_array_map(cosine_sim_pairwise_matrix,cosine_sim_pairwise_matrix_ij,starts,ends,starts_i,ends_i)
            percent_identity_mean_ij = zipped_results[1]
            percent_identity_mean= fill_array_map(percent_identity_mean,percent_identity_mean_ij,starts,ends,starts_i,ends_i)
            cosine_similarity_mean_ij = zipped_results[2]
            cosine_similarity_mean= fill_array_map(cosine_similarity_mean,cosine_similarity_mean_ij,starts,ends,starts_i,ends_i)
            kmers_cosine_similarity_ij = zipped_results[3]
            kmers_cosine_similarity_mean= fill_array_map(kmers_cosine_similarity,kmers_cosine_similarity_ij,starts,ends,starts_i,ends_i)
            kmers_pid_similarity_ij = zipped_results[4]
            kmers_pid_similarity= fill_array_map(kmers_pid_similarity,kmers_pid_similarity_ij,starts,ends,starts_i,ends_i)
            # kmers_matrix_cosine_diag_ij = zipped_results[5]
            # kmers_cosine_similarity_matrix_diag = fill_array_map(kmers_cosine_similarity_matrix_diag,kmers_matrix_cosine_diag_ij,starts,ends,starts_i,ends_i)

        end = time.time()
        print("Overall calculation time {}".format(str(datetime.timedelta(seconds=end - start))))
        #Highlight: Mirror values across the diagonal. Does not seem expensive
        #pid_pairwise_matrix = np.maximum(pid_pairwise_matrix, pid_pairwise_matrix.transpose(1,0,2,3))
        #kmers_cosine_similarity_matrix_diag = np.maximum(kmers_cosine_similarity_matrix_diag, kmers_cosine_similarity_matrix_diag.transpose(1,0,2,3,4))
        cosine_sim_pairwise_matrix = np.maximum(cosine_sim_pairwise_matrix, cosine_sim_pairwise_matrix.transpose(1,0,2,3))
        #positional_weights_kmers = importance_weight_kmers(kmers_cosine_similarity_matrix_diag,nkmers,ksize,max_len,array_mask,overlapping_kmers,batch_size) #TODO: Remove or?
        positional_weights = importance_weight(cosine_sim_pairwise_matrix,max_len,array_mask,batch_size,neighbours)
        percent_identity_mean = np.maximum(percent_identity_mean, percent_identity_mean.transpose())
        cosine_similarity_mean = np.maximum(cosine_similarity_mean, cosine_similarity_mean.transpose())
        kmers_pid_similarity = np.maximum(kmers_pid_similarity, kmers_pid_similarity.transpose())
        kmers_cosine_similarity_mean = np.maximum(kmers_cosine_similarity_mean, kmers_cosine_similarity_mean.transpose())

        np.save("{}/positional_weights.npy".format(storage_folder), positional_weights)
        np.save("{}/percent_identity_mean.npy".format(storage_folder),
                percent_identity_mean)
        np.save("{}/cosine_similarity_mean.npy".format(storage_folder),
                cosine_similarity_mean)
        np.save("{}/kmers_pid_similarity_{}ksize.npy".format(storage_folder, ksize),
                kmers_pid_similarity)
        np.save("{}/kmers_cosine_similarity_{}ksize.npy".format(storage_folder, ksize),
                kmers_cosine_similarity)

        similarity_results = SimilarityResults(positional_weights=np.ma.getdata(positional_weights),
                                               percent_identity_mean=np.ma.getdata(percent_identity_mean),
                                               cosine_similarity_mean=np.ma.getdata(cosine_similarity_mean),
                                               kmers_pid_similarity=np.ma.getdata(kmers_pid_similarity),
                                               kmers_cosine_similarity_mean=np.ma.getdata(kmers_cosine_similarity_mean))




        return similarity_results