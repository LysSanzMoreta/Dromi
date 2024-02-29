
import datetime,time
import numpy as np
import multiprocessing
import itertools
from sklearn.feature_selection import mutual_info_regression
import dromi
import dromi.utils as DromiUtils


def process_value(iterables_args,fixed_args):
    """"""

    i,j = iterables_args
    print("Calculating indexes i {} and j : {}".format(i,j))
    array,bins = fixed_args
    variable_x = array[:,i]
    variable_y = array[:,j]
    hist2d = np.histogram2d(variable_x, variable_y, bins)[0]

    return (hist2d,i,j)


def fill_array(histograms2d,hist2d, i,j):
    """"""
    histograms2d[i,j] = hist2d

    return histograms2d
def calculate_mutual_information(array,results_dir="",name = ""):

    """
    Calculates the normalized Mutual Information applied to continuous random variables

    I(X,Y) = H(X) + H(Y) -H(X,Y)
    i) Entropy of variable X
    H(X) = - \sum_i p(x_i) \log_2 p(x_i)
    ii) Joint entropy
    H(X,Y) = \sum_{x,y} p(x,y) \log_2 p(x,y)
    iii) Normalized Mutual information:
    NMI(X,Y) = \frac{2I(X,Y)}{H(X) + H(Y)}

    :param array : Matrix nxm
    :param str results_dir: Path where to save the results
    NOTES:
        -References:
            Guo, Xue, Hu Zhang, and Tianhai Tian. "Development of stock correlation networks using mutual information and financial big data." PloS one 13.4 (2018): e0195941.
        -Following: https://medium.com/@bass.dolg/computing-mutual-information-matrix-with-python-6ced9169bcb1

    """

    start = time.time()
    ndata = array.shape[0]
    i = np.repeat(np.arange(ndata),ndata).tolist()
    j = np.tile(np.arange(ndata), ndata).tolist()
    args_iterables = {"i_idx":i,"j_idx":j}
    bins= int((array.shape[0]/5)**.5)
    if bins < 1: #small arrays
        bins = 1
    args_fixed = array,bins
    histograms2d = np.zeros((array.shape[0],array.shape[-1],bins,bins),dtype=np.float16)

    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
        results = DromiUtils.RunParallel(args_iterables, args_fixed,process_value).outer_loop(pool)
        zipped_results = list(zip(*results))
        hist2d_list = zipped_results[0]
        i_idx = zipped_results[1] #this is just in case we need to keep track of more advanced indexing
        j_idx = zipped_results[2]

    histograms2d_result = list(map(lambda hist2d,i,j: fill_array(histograms2d,hist2d, i,j), hist2d_list,i_idx,j_idx ))[0] #this can be moved inside process_value function, since it is very simple indexing in this case

    probs = histograms2d_result / ndata + 1e-100
    joint_entropies = -(probs * np.log2(probs)).sum((2, 3))  #H(x,y)
    entropies = joint_entropies.diagonal() #H(x) , H(y)
    sum_entropies = entropies + entropies.T
    mi_matrix = sum_entropies - joint_entropies

    norm_mi_matrix = mi_matrix * 2 / sum_entropies
    end = time.time()

    print("Overall calculation time {}".format(str(datetime.timedelta(seconds=end - start))))

    results_dir = "" if not results_dir else f"{results_dir}/"
    name = "" if not name else f"_{name}"

    np.save("{}Mutual_information{}.npy".format(results_dir, name), mi_matrix)
    np.save("{}Mutual_information_normalized{}.npy".format(results_dir,name),norm_mi_matrix)



