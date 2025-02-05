"""
=======================
2023: Lys Sanz Moreta
Dromi: Python package for parallel computation of similarity measures among vector-encoded sequences
=======================
"""
import os, sys, argparse
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import RawTextHelpFormatter
import numpy as np
from collections import namedtuple
from typing import Union

local_repository = True
script_dir = os.path.dirname(os.path.abspath(__file__))

if local_repository:
    sys.path.insert(1, "{}/dromi/src".format(script_dir))
    import dromi
else:  # pip installed module
    import dromi
import dromi.utils as DromiUtils
import dromi.similarities as DromiSimilarities
import dromi.mutual_information as DromiMI

print("Loading dromi module from {}".format(dromi.__file__))


def plot_heatmap(array: np.ndarray, title: str, file_name: str):
    """Plot heatmap of array
    :param array: Numpy array
    :param title: Plot title
    :param file_name"""
    print("Visualizing heatmap...")
    fig = plt.figure(figsize=(20, 20))
    ax = sns.heatmap(array, cmap='RdYlGn_r', yticklabels=False, xticklabels=False)
    ax.collections[0].set_clim(0, 1)
    plt.title(title, fontsize=20)
    plt.savefig(file_name)
    plt.clf()
    plt.close(fig)


def select_plots(args: argparse.Namespace, results: namedtuple, storage_folder, suffix: str):
    """Performs different plots according to the selected arguments in the cli"""
    if args.metric in ["cosine", "all"]:
        plot_heatmap(results.cosine_similarity_mean, "HEATMAP Cosine similarity mean",
                     "{}/HEATMAP_cosine_similarity_mean{}".format(storage_folder, suffix))
        if args.calculate_kmers:
            plot_heatmap(results.kmers_cosine_similarity_mean, "HEATMAP Kmers cosine similarity mean",
                         "{}/HEATMAP_kmers_cosine_similarity_mean{}".format(storage_folder, suffix))
    if args.metric in ["pairwise", "all"]:
        plot_heatmap(results.percent_identity_mean, "HEATMAP Percent Identity mean",
                     "{}/HEATMAP_pecent_id_mean{}".format(storage_folder, suffix))
        if args.calculate_kmers:
            plot_heatmap(results.kmers_pid_similarity, "HEATMAP Kmers percent identity mean",
                         "{}/HEATMAP_kmers_pid_similarity{}".format(storage_folder, suffix))

    if args.metric == "cosine" and args.calculate_positional_weights:
        plot_heatmap(results.positional_weights, "HEATMAP Positional weights",
                     "{}/HEATMAP_positional_weights{}".format(storage_folder, suffix))


def calculate_similarities_options(array: np.ndarray, max_len: Union[int, float], array_mask: np.ndarray,
                                   storage_folder: str, args: argparse.Namespace):
    if args.runtime == "ram":
        results, final_time = DromiSimilarities.calculate_similarities(array, max_len, array_mask, storage_folder,
                                                                       batch_size=1,
                                                                       ksize=3,
                                                                       neighbours=1,
                                                                       metric=args.metric,
                                                                       calculate_kmers=args.calculate_kmers,
                                                                       calculate_positional_weights=args.calculate_positional_weights)

    elif args.runtime == "disk":
        results, final_time = DromiSimilarities.calculate_similarities_ondisk(array, max_len, array_mask,
                                                                              storage_folder,
                                                                              batch_size=1,
                                                                              ksize=3,
                                                                              neighbours=1,
                                                                              metric=args.metric,
                                                                              calculate_kmers=args.calculate_kmers,
                                                                              calculate_positional_weights=args.calculate_positional_weights)
    elif args.runtime == "cuda_cpu":
        results, final_time = DromiSimilarities.calculate_similarities_cuda_cpu(array, max_len, array_mask,
                                                                                storage_folder,
                                                                                batch_size=1,
                                                                                ksize=3,
                                                                                neighbours=1,
                                                                                metric=args.metric,
                                                                                calculate_kmers=args.calculate_kmers,
                                                                                calculate_positional_weights=args.calculate_positional_weights)

    c = results.cosine_similarity_mean
    print(c)
    #
    # exit()

    return results


def example_blosum_encoded_sequences(unique_characters: Union[int, float] = 21, random_seqs: bool = False):
    """The similarity computations are performed excluding self similarity. The current position is compared to the other positions in the same site.
    NOTE: I have only implemented similarity matrix with paddings at the end, if requested I might look into paddings with other distributions
    """
    if random_seqs:
        random_result = DromiUtils.SequenceRandomGeneration(["".join(["A"] * 40)] * 300, 50, "no_padding").run()
        seqs, sequences_padded = zip(*random_result)
        max_len = len(max(seqs, key=len))

    else:
        # seqs = ["AHPDYRMPIL"] * 1000
        seqs = ["AHPDYRM",
                "AHPHYRM",
                "AKPDYRM",
                "AHPDYRM",
                "AHPDYRM",
                "FYRA",
                "MRSTVI"]
        # seqs = [
        #     "RGICWMLV",
        #     "RGICWMLV",
        #     "RGVCWMLV",
        #     "RGVCWMLV",
        #     "RGACWMLV",
        #     "RGACFMLV",
        #     "RGLCYMLV",
        #     "RGLCYMLV",
        #     "RGICYMLV",
        #     "RGICYMLV",
        # ]
        max_len = len(max(seqs, key=len))

        padding_result = DromiUtils.SequencePadding(seqs, max_len, method="ends", shuffle=False).run()
        sequences, sequences_padded = zip(*padding_result)  # unpack list of tuples onto 2 lists

    blosum_array, blosum_dict, blosum_array_dict = DromiUtils.create_blosum(unique_characters, "BLOSUM62",
                                                                            zero_characters=["#"],
                                                                            include_zero_characters=True)

    aa_dict = DromiUtils.aminoacid_names_dict(21, zero_characters=["#"])
    sequences_array = np.array(sequences_padded)
    sequences_int = np.vectorize(aa_dict.get)(sequences_array)
    sequences_blosum = np.vectorize(blosum_array_dict.get, signature='()->(n)')(sequences_int)
    sequences_mask = sequences_int.astype(bool)
    storage_folder = "{}".format(script_dir)
    start = time.time()

    results = calculate_similarities_options(sequences_blosum, max_len, sequences_mask, f"{storage_folder}", args)

    stop = time.time()
    print("Finished in {}".format(str(datetime.timedelta(seconds=stop - start))))
    # TODO: Positional weights are returned also when rgs.metric == <pairwise>
    # TODO: Test runtime with and without deleting objects and gc.collect
    select_plots(args, results, storage_folder, f"_{args.runtime}")


def example_mutual_information():
    """On disk computation of Mutual information between continuous random variables"""

    gene_expression_matrix = np.random.rand(2000, 503)
    mi_results = DromiMI.calculate_mutual_information(gene_expression_matrix, bins=5)

    results_dir = ""
    name = ""
    results_dir = "" if not results_dir else f"{results_dir}/"
    name = "" if not name else f"_{name}"
    np.save("{}Mutual_information{}.npy".format(results_dir, name), mi_results["mutual_information"])
    np.save("{}Mutual_information_normalized{}.npy".format(results_dir, name),
            mi_results["normalized_mutual_information"])


def example_vector_encoded_sequences():  # TODO: Refactor
    # sequences = np.load("data/protein_subset.npy")
    sequences = np.load("data/full_set.npy")

    max_len = 75
    storage_folder = "{}".format(script_dir)
    start = time.time()

    results = calculate_similarities_options(sequences, max_len, None, storage_folder, args)

    stop = time.time()
    print("Finished in {}".format(str(datetime.timedelta(seconds=stop - start))))
    select_plots(args, results, storage_folder)


def parse_args(parser):
    parser.add_argument('-analysis', type=str, nargs='?', default="similarities",
                        help='Whether to calculate sequence similarities (cosine, percent identity) or mutual information'
                             '<similarities> \n'
                             '<mutualinfo>')

    parser.add_argument('-runtime', type=str, nargs='?', default="ram",
                        help='How to compute/store the calculations'
                             '<ram>: Python-native The chunked results are computed and accumulated on RAM \n'
                             '<disk>: The results arrays are initialized on disk and filled up with the chunked computations made by the RAM'
                             '<cuda-cpu>: Splits and stores the array into binary files that are used by a C-version of the cosine similarity function '
                        )
    parser.add_argument('-metric', type=str, nargs='?', default="cosine",
                        help='Type of sequence similarities metric (cosine, pairwise, use when args.analysis == <similarities>'
                             '<cosine> \n'
                             '<pairwise>: Percent identity \n'
                             '<all>: calculates both cosine and percent identity metrics')

    parser.add_argument('-calculate_kmers', action=argparse.BooleanOptionalAction, default=False,
                        help='Add calculation of kmers similarity, use when args.analysis == <similarities>. Example: If args.metric is <cosine> then it will calculate the kmers cosine similarity'
                             '<True>\n'
                             '<False>')

    parser.add_argument('-calculate_positional_weights', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Add calculation of positional weights, use when args.analysis == <similarities>.'
                             '<True>\n'
                             '<False>')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dromi args", formatter_class=RawTextHelpFormatter)

    args = parse_args(parser)  # This was separated for the GUI program with Gooey, eventually can be re-merged
    if args.analysis == "similarities":
        example_blosum_encoded_sequences()
        # example_vector_encoded_sequences()
    elif args.analysis == "mutualinfo":
        example_mutual_information()

    """
    disk batch 3
    
[[1.         0.91699219 0.91503906 1.         0.99989832 0.35865713 0.34929686]
 [0.91699219 1.         0.83203125 0.91699219 0.91705206 0.29680633 0.29621005]
 [0.91503906 0.83203125 1.         0.91503906 0.91523214 0.2172862 0.41001359]
 [1.         0.91699219 0.91503906 1.         0.99989832 0.35865713 0.34929686]
 [0.99989832 0.91705206 0.91523214 0.99989832 1.         0.35864258 0.34936523]
 [0.35865713 0.29680633 0.2172862  0.35865713 0.35864258 0.99951172 0.4453125 ]
 [0.34929686 0.29621005 0.41001359 0.34929686 0.34936523 0.4453125 1.        ]]
  
  
  ram batch 3
    
[[1.     0.917  0.9155 1.     1.     0.3586 0.3494]
 [0.917  1.     0.8325 0.917  0.917  0.2966 0.2961]
 [0.9155 0.8325 1.     0.9155 0.9155 0.2173 0.4102]
 [1.     0.917  0.9155 1.     1.     0.3586 0.3494]
 [1.     0.917  0.9155 1.     1.     0.3586 0.3494]
 [0.3586 0.2966 0.2173 0.3586 0.3586 1.     0.4453]
 [0.3494 0.2961 0.4102 0.3494 0.3494 0.4453 1.    ]]
 

  disk batch 1
  
[[1.         0.91699219 0.91503906 1.         1.         0.35864258 0.34936523]
 [0.91699219 1.         1.         0.91699219 0.91699219 0.296875   0.29614258]
 [0.91503906 1.         0.99951172 0.91503906 1.         0.21728516 0.41015625]
 [1.         0.91699219 0.91503906 1.         1.         0.35864258 0.        ]
 [1.         0.91699219 1.         1.         1.         0.35864258 0.34936523]
 [0.35864258 0.296875   0.21728516 0.35864258 0.35864258 0.99951172 0.4453125 ]
 [0.34936523 0.29614258 0.41015625 0.         0.34936523 0.4453125  1.        ]]
  
  
  ram batch 1
  
  [[1.     0.917  0.9155 1.     1.     0.3586 0.3494]
 [0.917  1.     1.     0.917  0.917  0.2966 0.2961]
 [0.9155 1.     1.     0.9155 1.     0.2173 0.4102]
 [1.     0.917  0.9155 1.     1.     0.3586 0.    ]
 [1.     0.917  1.     1.     1.     0.3586 0.3494]
 [0.3586 0.2966 0.2173 0.3586 0.3586 1.     0.4453]
 [0.3494 0.2961 0.4102 0.     0.3494 0.4453 1.    ]]
 
 
 ram batch 2
 
 [[1.     0.917  0.9155 1.     1.     0.3586 0.3494]
 [0.917  1.     0.8325 0.917  0.917  0.2966 0.2961]
 [0.9155 0.8325 1.     0.9155 0.9155 0.2173 0.4102]
 [1.     0.917  0.9155 1.     1.     0.     0.3494]
 [1.     0.917  0.9155 1.     1.     0.3586 0.    ]
 [0.3586 0.2966 0.2173 0.     0.3586 1.     0.4453]
 [0.3494 0.2961 0.4102 0.3494 0.     0.4453 1.    ]]
 
  
    
    """
