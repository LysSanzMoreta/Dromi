import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import  cosine_similarity as cssk
from numba import jit
import sparse
def cosine_similarity_old(a, b, correlation_matrix=False, parallel=False):  # TODO: import from utils?
    """Calculates the cosine similarity between 2 arrays.
    :param numpy array a: (max_len,aa_types) or (num_seq,max_len, aa_types)
    :param numpy array b: (max_len,aa_types) or (num_seq,max_len, aa_types)
    :param bool:Calculate matrix correlation(as in numpy coorcoef)"""

    n_a = a.shape[0]
    n_b = b.shape[0]
    diff_sizes = False
    if n_a != n_b:
        dummy_row = np.zeros((np.abs(n_a - n_b),) + a.shape[1:])
        diff_sizes = True
        if n_a < n_b:
            a = np.concatenate((a, dummy_row), axis=0)
        else:
            b = np.concatenate((b, dummy_row), axis=0)

    if np.ndim(a) == 1:
        num = np.dot(a, b)
        p1 = np.sqrt(np.sum(a ** 2))  # equivalent to p1 = np.linalg.norm(a)
        p2 = np.sqrt(np.sum(b ** 2))
        p1_p2 = p1 * p2

        p1_p2 = np.where(p1_p2 == 0, np.finfo(float).eps, p1_p2)  # avoid zero division issues
        # cosine_sim = num / (p1 * p2)
        cosine_sim = num / (p1_p2)
        return cosine_sim

    elif np.ndim(a) == 2:
        if correlation_matrix:
            b = b - b.mean(axis=1)[:, None]
            a = a - a.mean(axis=1)[:, None]

        num = np.dot(a, b.T)  # [seq_len,21]@[21,seq_len] = [seq_len,seq_len]
        p1 = np.sqrt(np.sum(a ** 2, axis=1))[:, None]  # + 1e-10#[seq_len,1] #equivalent to np.linalg.norm(a,axis=1)
        # p1 = np.linalg.norm(a,axis=1)[:,None]
        p2 = np.sqrt(np.sum(b ** 2, axis=1))[None, :]  # [1,seq_len]
        # p2 = np.linalg.norm(b,axis=1)[None,:]
        p1_p2 = p1 * p2
        p1_p2 = np.where(p1_p2 == 0, np.finfo(float).eps, p1_p2)  # avoid zero division issues
        # cosine_sim = num / (p1 * p2)
        cosine_sim = num / (p1_p2)

        if diff_sizes:  # remove the dummy creation that was made avoid shape conflicts
            remove = np.abs(n_a - n_b)
            if n_a < n_b:
                cosine_sim = cosine_sim[:-remove]
            else:
                cosine_sim = cosine_sim[:, :-remove]
        if parallel:
            return cosine_sim[None, :]
        else:
            return cosine_sim
    else:  # TODO: use elipsis for general approach?
        if correlation_matrix:
            b = b - b.mean(axis=2)[:, :, None]
            a = a - a.mean(axis=2)[:, :, None]
        num = np.matmul(a[:, None], np.transpose(b, (0, 2, 1))[None, :])  # [n,n,seq_len,seq_len]
        p1 = np.sqrt(np.sum(a ** 2, axis=2))[:, :, None] + 1e-10  # Equivalent to np.linalg.norm(a,axis=2)[:,:,None]
        p2 = np.sqrt(np.sum(b ** 2, axis=2))[:, None, :] + 1e-10  # Equivalent to np.linalg.norm(b,axis=2)[:,None,:]

        cosine_sim = num / (p1[:, None] * p2[None, :])

        if diff_sizes:  # remove the dummy creation that was made avoid shape conflicts
            remove = np.abs(n_a - n_b)
            if n_a < n_b:
                cosine_sim = cosine_sim[:-remove]
            else:
                cosine_sim = cosine_sim[:, :-remove]

        return cosine_sim


def cosine_similarity(a, b, correlation_matrix=False, parallel=False):
    """Calculates the cosine similarity between 2 arrays.
    :param numpy array a: (max_len,aa_types) or (num_seq,max_len, aa_types)
    :param numpy array b: (max_len,aa_types) or (num_seq,max_len, aa_types)
    :param bool:Calculate matrix correlation(as in numpy coorcoef)


    NOTES: https://stackoverflow.com/questions/17627219/whats-the-fastest-way-in-python-to-calculate-cosine-similarity-given-sparse-mat
    Benchmrks: Cyton, python, C: https://ashvardanian.com/posts/python-c-assembly-comparison/
    """

    n_a = a.shape[0]
    n_b = b.shape[0]
    diff_sizes = False
    ndim = a.ndim if isinstance(a,np.ndarray) or isinstance(a,sparse._coo.core.COO) else a.data.ndim

    if n_a != n_b:
        dummy_row = np.zeros((np.abs(n_a - n_b),) + a.shape[1:])
        diff_sizes = True
        if n_a < n_b:
            a = np.concatenate((a, dummy_row), axis=0)
        else:
            b = np.concatenate((b, dummy_row), axis=0)

    if ndim == 1: # TODO: https://stackoverflow.com/questions/17627219/whats-the-fastest-way-in-python-to-calculate-cosine-similarity-given-sparse-mat

        squared_a, squared_b = a.multiply(a), b.multiply(b)
        #TODO: Weird that it is not sum over axis 0
        sqrt_sum_squared_rows_a, sqrt_sum_squared_rows_b = np.array(np.sqrt(squared_a.sum(axis=1)))[:, 0],  np.array(np.sqrt(squared_b.sum(axis=1)))[:, 0]
        (row_indices_a, col_indices_a), (row_indices_b, col_indices_b) = a.nonzero(), b.nonzero()
        a.data /= sqrt_sum_squared_rows_a[row_indices_a]
        b.data /= sqrt_sum_squared_rows_b[row_indices_b]
        return a.dot(b.T)



    elif ndim == 2:


        squared_a, squared_b = a.multiply(a), b.multiply(b)
        # TODO: Weird that it is not sum over axis 0
        sqrt_sum_squared_rows_a, sqrt_sum_squared_rows_b = np.array(np.sqrt(squared_a.sum(axis=2)))[:, 0],  np.array(np.sqrt(squared_b.sum(axis=2)))[:, 0]
        (row_indices_a, col_indices_a), (row_indices_b, col_indices_b) = a.nonzero(), b.nonzero()
        a.data /= sqrt_sum_squared_rows_a[row_indices_a]
        b.data /= sqrt_sum_squared_rows_b[row_indices_b]
        cosine_sim = a.dot(b.T)
        if diff_sizes:  # remove the dummy creation that was made avoid shape conflicts
            remove = np.abs(n_a - n_b)
            if n_a < n_b:
                cosine_sim = cosine_sim[:-remove]
            else:
                cosine_sim = cosine_sim[:, :-remove]
        if parallel:
            return cosine_sim[None, :]
        else:
            return cosine_sim
    else:  # TODO: use elipsis for general approach?
        if correlation_matrix:
            b = b - b.mean(axis=2)[:, :, None]
            a = a - a.mean(axis=2)[:, :, None]

        squared_a, squared_b = np.multiply(a, a), np.multiply(b, b)


        w = np.sqrt(squared_a.sum(axis=2))[:,0]


        sqrt_sum_squared_rows_a, sqrt_sum_squared_rows_b = np.sqrt(squared_a.sum(axis=2))[:, 0],  np.sqrt(squared_b.sum(axis=2))[:, 0]
        (row_indices_a, col_indices_a, _), (row_indices_b, col_indices_b,_) = a.nonzero(), b.nonzero()
        a.data /= sqrt_sum_squared_rows_a[row_indices_a]
        b.data /= sqrt_sum_squared_rows_b[row_indices_b]
        cosine_sim = a.dot(b.T)
        print(cosine_sim)
        if diff_sizes:  # remove the dummy creation that was made avoid shape conflicts
            remove = np.abs(n_a - n_b)
            if n_a < n_b:
                cosine_sim = cosine_sim[:-remove]
            else:
                cosine_sim = cosine_sim[:, :-remove]

        return cosine_sim

np.random.seed(0)
a = np.random.rand(10,)
b = a + 1

# r = cssk(a.reshape(1, -1),b.reshape(1, -1))
# print("sklearn cosine sim dense")
# print(r)
# print("--------------------------")
# r = cosine_similarity_old(a,b)
# print("Old cosine similarity")
# print(r)
# print("--------------------------")
# a= csr_matrix(a)
# b = csr_matrix(b)
# r = cssk(a,b)
# print("sklearn cosine sim sparse")
# print(r)
# print("--------------------------")
# print("New cosine similarity")
# r = cosine_similarity(a,b)
# print(r)
# print("--------------------------")
# exit()

n = 2
a = np.random.rand(n,10)
b = np.random.rand(n,10) +  1

# r = cssk(a,b)
# print("sklearn cosine sim dense")
# print(r)
# print("--------------------------")
#
# r = cosine_similarity_old(a,b)
# print("Old cosine similarity")
# print(r)
# print("--------------------------")
# a= csr_matrix(a)
# b = csr_matrix(b)
# r = cssk(a,b)
# print("sklearn cosine sim sparse")
# print(r)
# print("--------------------------")
#
# print("New cosine similarity")
# r = cosine_similarity(a,b)
# print(r)
# print("--------------------------")
# exit()

n = 2
a = np.random.rand(n,5,10)
b = np.random.rand(n,5,10) +  1


r = cosine_similarity_old(a,b)
print("Old cosine similarity")
print(r)
print("--------------------------")
a= sparse.COO(a)
b = sparse.COO(b)

print("New cosine similarity")
r = cosine_similarity(a,b)
print(r)
print("--------------------------")
